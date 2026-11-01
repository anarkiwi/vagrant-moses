//! `vagrant_moses._core`: the native cycle-exact C64 for Python.
//!
//! State lives in `UnsafeCell`s behind frozen pyclasses. Every Python entry point holds
//! the GIL; `run` additionally claims the machine for the running thread and releases
//! the GIL, re-acquiring it only to call SID devices, the event sink and the hooks. While
//! it runs, other threads are refused and only a hook (which is handed the core the run
//! loop paused) may touch the machine.

use std::cell::UnsafeCell;
use std::ffi::{c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use jennings_core::AddrSet;
use moses_core::lines::NAMES;
use moses_core::machine::{K_ABSX, K_ABSY, K_BRANCH, K_INDY, K_NONE, K_RMW};
use moses_core::reu::{geometry, REGS, SIZES};
use moses_core::{smc, Config, Core, Host, Layout, Recorder, SidDev, SidShared, Watch};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyFrozenSet, PySlice, PyTuple};

thread_local!(static TOKEN: u8 = const { 0 });

fn token() -> usize {
    TOKEN.with(|t| t as *const u8 as usize)
}

/// Idle (0) or the token of the thread running the machine.
#[derive(Default)]
struct Owner(AtomicUsize);

impl Owner {
    fn check(&self) -> PyResult<()> {
        match self.0.load(Ordering::Acquire) {
            0 => Ok(()),
            t if t == token() => Ok(()),
            _ => Err(PyRuntimeError::new_err(
                "the machine is running on another thread",
            )),
        }
    }

    fn here(&self) -> bool {
        self.0.load(Ordering::Acquire) == token()
    }

    fn claim(&self) -> PyResult<Claim<'_>> {
        self.0
            .compare_exchange(0, token(), Ordering::AcqRel, Ordering::Acquire)
            .map(|_| Claim(self))
            .map_err(|_| PyRuntimeError::new_err("the machine is already running"))
    }
}

struct Claim<'a>(&'a Owner);

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        self.0 .0.store(0, Ordering::Release);
    }
}

/// `*mut Core` the detached closure may carry (the claim keeps it exclusive).
struct CorePtr(*mut Core);
// SAFETY: the pointee outlives the detached run, which the caller joins.
unsafe impl Send for CorePtr {}

#[derive(Default)]
struct Hooks {
    trap: Option<Py<PyAny>>,
    decode: Option<Py<PyAny>>,
    port: Option<Py<PyAny>>,
}

/// Singleton for "watch every address".
#[pyclass(module = "vagrant_moses._core", name = "WatchAll", frozen)]
struct WatchAll;

#[pymethods]
impl WatchAll {
    fn __contains__(&self, _addr: i64) -> bool {
        true
    }

    fn __repr__(&self) -> &'static str {
        "WATCH_ALL"
    }
}

static WATCH_ALL: PyOnceLock<Py<WatchAll>> = PyOnceLock::new();

fn watch_all(py: Python<'_>) -> PyResult<&Py<WatchAll>> {
    WATCH_ALL.get_or_try_init(py, || Py::new(py, WatchAll))
}

/// The Python side of a run: devices, sink and hooks, called with the GIL re-acquired.
struct PyHost<'a> {
    devs: &'a [HostDev],
    sink: Option<&'a Py<PyAny>>,
    hooks: Hooks,
    me: Option<Py<Machine>>,
    active: &'a AtomicPtr<Core>,
    err: Option<PyErr>,
}

/// A SID base's Python side: a device with `read`/`write`, or a `NativeSid`'s `on_read`.
struct HostDev {
    obj: Py<PyAny>,
    on_read: bool,
}

impl PyHost<'_> {
    fn fail(&mut self, e: PyErr) {
        self.err.get_or_insert(e);
    }

    fn hook(&mut self, core: &mut Core, cb: Option<Py<PyAny>>) -> bool {
        let (Some(cb), Some(me)) = (cb, self.me.as_ref()) else {
            return false;
        };
        if self.err.is_some() {
            return false;
        }
        self.active.store(core, Ordering::Release);
        let r = Python::attach(|py| {
            cb.call1(py, (me.clone_ref(py),))
                .and_then(|v| v.is_truthy(py))
        });
        self.active.store(ptr::null_mut(), Ordering::Release);
        r.unwrap_or_else(|e| {
            self.fail(e);
            false
        })
    }
}

impl Host for PyHost<'_> {
    fn sid_read(&mut self, dev: usize, cycle: i64, reg: u8) -> u8 {
        if self.err.is_some() {
            return 0;
        }
        let r = Python::attach(|py| {
            let d = &self.devs[dev];
            if d.on_read {
                d.obj.call1(py, (cycle, reg))
            } else {
                d.obj.call_method1(py, "read", (cycle, reg))
            }
            .and_then(|v| v.extract::<i64>(py))
        });
        r.map(|v| v as u8).unwrap_or_else(|e| {
            self.fail(e);
            0
        })
    }

    fn sid_write(&mut self, dev: usize, cycle: i64, reg: u8, value: u8) {
        if self.err.is_some() {
            return;
        }
        if let Err(e) = Python::attach(|py| {
            self.devs[dev]
                .obj
                .call_method1(py, "write", (cycle, reg, value))
                .map(drop)
        }) {
            self.fail(e);
        }
    }

    fn event(&mut self, cycle: i64, name: u8) {
        let (Some(sink), None) = (self.sink, &self.err) else {
            return;
        };
        if let Err(e) = Python::attach(|py| sink.call1(py, (cycle, NAMES[name as usize])).map(drop))
        {
            self.fail(e);
        }
    }

    fn trap(&mut self, core: &mut Core) -> bool {
        let cb = Python::attach(|py| self.hooks.trap.as_ref().map(|c| c.clone_ref(py)));
        self.hook(core, cb)
    }

    fn decode(&mut self, core: &mut Core) {
        let cb = Python::attach(|py| self.hooks.decode.as_ref().map(|c| c.clone_ref(py)));
        self.hook(core, cb);
    }

    fn port_write(&mut self, core: &mut Core) {
        let cb = Python::attach(|py| self.hooks.port.as_ref().map(|c| c.clone_ref(py)));
        self.hook(core, cb);
    }

    fn failed(&self) -> bool {
        self.err.is_some()
    }
}

fn byte(v: i64) -> PyResult<u8> {
    u8::try_from(v).map_err(|_| PyValueError::new_err("byte must be in range(0, 256)"))
}

fn octets(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(buf) = PyBuffer::<u8>::get(obj) {
        return buf.to_vec(obj.py());
    }
    obj.try_iter()?
        .map(|v| v.and_then(|v| v.extract::<i64>()).and_then(byte))
        .collect()
}

fn image(data: &[u8]) -> PyResult<Box<[u8; 0x10000]>> {
    <Box<[u8; 0x10000]>>::try_from(data.to_vec().into_boxed_slice())
        .map_err(|_| PyValueError::new_err("a ROM or RAM image is 65536 bytes"))
}

fn layout(item: &Bound<'_, PyAny>) -> PyResult<Layout> {
    let (ents, exec_at, kind, base, branch): (Vec<u8>, i32, u8, i32, Option<(u8, bool)>) =
        item.extract()?;
    let kind = match kind {
        K_NONE | K_RMW | K_ABSX | K_ABSY | K_INDY | K_BRANCH => kind,
        _ => return Err(PyValueError::new_err("unknown layout kind")),
    };
    Ok(Layout {
        ents,
        exec_at,
        kind,
        base,
        branch,
    })
}

/// The cycle-exact C64 (`vagrant_moses.machine.Machine`) running natively.
#[pyclass(module = "vagrant_moses._core", name = "Machine", frozen, subclass)]
pub struct Machine {
    core: UnsafeCell<Box<Core>>,
    devs: Vec<HostDev>,
    logged: Vec<Py<NativeSid>>,
    sids: Py<PyDict>,
    sink: Option<Py<PyAny>>,
    hooks: UnsafeCell<Hooks>,
    owner: Owner,
    active: AtomicPtr<Core>,
    ntsc: bool,
    views: [OnceLock<Py<Memory>>; 6],
}

// SAFETY: every access holds the GIL; while a run is detached, `owner` refuses all threads
// but the runner's, whose hooks reach the core only through `active`.
unsafe impl Sync for Machine {}

impl Machine {
    #[allow(clippy::mut_from_ref)]
    fn core(&self) -> PyResult<&mut Core> {
        self.owner.check()?;
        if self.owner.here() {
            let p = self.active.load(Ordering::Acquire);
            if p.is_null() {
                return Err(PyRuntimeError::new_err("the machine is running"));
            }
            // SAFETY: a hook is running and `p` is the core the run loop handed it.
            return Ok(unsafe { &mut *p });
        }
        // SAFETY: idle machine, GIL held.
        Ok(unsafe { &mut **self.core.get() })
    }

    #[allow(clippy::mut_from_ref)]
    fn hooks(&self) -> &mut Hooks {
        // SAFETY: GIL held; callers never hold two borrows at once.
        unsafe { &mut *self.hooks.get() }
    }

    /// The one view of memory `which` (0 RAM, 1 colour RAM, 2-4 BASIC/CHARGEN/KERNAL, 5 REU).
    fn view(slf: &Bound<'_, Self>, which: u8) -> PyResult<Py<Memory>> {
        let cell = &slf.get().views[which as usize];
        if let Some(v) = cell.get() {
            return Ok(v.clone_ref(slf.py()));
        }
        let v = Py::new(
            slf.py(),
            Memory {
                machine: slf.clone().unbind(),
                which,
            },
        )?;
        Ok(cell.get_or_init(|| v).clone_ref(slf.py()))
    }

    fn host(&self) -> PyHost<'_> {
        PyHost {
            devs: &self.devs,
            sink: self.sink.as_ref(),
            hooks: Hooks::default(),
            me: None,
            active: &self.active,
            err: None,
        }
    }

    #[allow(clippy::mut_from_ref)]
    fn rec<'a>(&self, core: &'a mut Core) -> PyResult<&'a mut Recorder> {
        core.s
            .rec
            .as_deref_mut()
            .ok_or_else(|| PyRuntimeError::new_err("the SMC recorder is not armed"))
    }

    /// The hook a port write dispatches to, once the write is made.
    fn port_hook(slf: &Bound<'_, Self>, core: &mut Core) -> PyResult<()> {
        if !std::mem::take(&mut core.s.port_dirty) || !core.armed || !core.port_set {
            return Ok(());
        }
        let cb = slf
            .get()
            .hooks()
            .port
            .as_ref()
            .map(|c| c.clone_ref(slf.py()));
        match cb {
            Some(cb) => cb.call1(slf.py(), (slf,)).map(drop),
            None => Ok(()),
        }
    }
}

#[pymethods]
impl Machine {
    #[new]
    #[pyo3(signature = (ntsc, sids, cia_new, on_event, layouts, ram, roms, lines, cycles_per_line, reu=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        ntsc: bool,
        sids: Bound<'_, PyDict>,
        cia_new: bool,
        on_event: Option<Py<PyAny>>,
        layouts: &Bound<'_, PyAny>,
        ram: &[u8],
        roms: (Vec<u8>, Vec<u8>, Vec<u8>),
        lines: i32,
        cycles_per_line: i32,
        reu: Option<u32>,
    ) -> PyResult<Self> {
        if let Some(kb) = reu.filter(|&kb| geometry(kb).is_none()) {
            return Err(PyValueError::new_err(format!(
                "REU size must be one of {SIZES:?} KiB, got {kb}"
            )));
        }
        let mut devs = Vec::new();
        let mut map = Vec::new();
        let mut logged = Vec::new();
        for (k, v) in sids.iter() {
            let base = k.extract::<u32>()?;
            if let Ok(sid) = v.cast::<NativeSid>() {
                let sid = sid.clone();
                map.push((base, SidDev::Logged(sid.get().shared.clone(), devs.len())));
                devs.push(HostDev {
                    obj: sid.get().on_read.clone_ref(sid.py()),
                    on_read: true,
                });
                logged.push(sid.unbind());
            } else {
                map.push((base, SidDev::Host(devs.len())));
                devs.push(HostDev {
                    obj: v.unbind(),
                    on_read: false,
                });
            }
        }
        let layouts = layouts
            .try_iter()?
            .map(|l| l.and_then(|l| layout(&l)))
            .collect::<PyResult<Vec<_>>>()?;
        let cfg = Config {
            ntsc,
            cia_new,
            ram: image(ram)?,
            roms: [image(&roms.0)?, image(&roms.1)?, image(&roms.2)?],
            layouts,
            sids: map,
            custom_sink: on_event.is_some(),
            lines,
            cycles_per_line,
            reu,
        };
        Ok(Machine {
            core: UnsafeCell::new(Box::new(Core::new(cfg))),
            devs,
            logged,
            sids: sids.unbind(),
            sink: on_event,
            hooks: UnsafeCell::new(Hooks::default()),
            owner: Owner::default(),
            active: AtomicPtr::new(ptr::null_mut()),
            ntsc,
            views: Default::default(),
        })
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn WATCH_ALL(py: Python<'_>) -> PyResult<Py<WatchAll>> {
        Ok(watch_all(py)?.clone_ref(py))
    }

    #[getter]
    fn ntsc(&self) -> bool {
        self.ntsc
    }

    #[getter]
    fn sids<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.sids.bind(py).clone()
    }

    #[getter]
    fn t(&self) -> PyResult<i64> {
        Ok(self.core()?.t)
    }

    #[getter]
    fn cycle(&self) -> PyResult<i64> {
        Ok(self.core()?.t)
    }

    #[getter]
    fn jammed(&self) -> PyResult<bool> {
        Ok(self.core()?.jammed)
    }

    #[getter]
    fn armed(&self) -> PyResult<bool> {
        Ok(self.core()?.armed)
    }

    #[getter]
    fn stop(&self) -> PyResult<bool> {
        Ok(self.core()?.stop)
    }

    #[setter]
    fn set_stop(&self, v: bool) -> PyResult<()> {
        self.core()?.stop = v;
        Ok(())
    }

    #[getter]
    fn port_data(&self) -> PyResult<u8> {
        Ok(self.core()?.s.port_data)
    }

    #[getter]
    fn port_dir(&self) -> PyResult<u8> {
        Ok(self.core()?.s.port_dir)
    }

    #[getter]
    fn io(&self) -> PyResult<bool> {
        Ok(self.core()?.s.io)
    }

    #[getter]
    fn bits(&self) -> PyResult<i32> {
        Ok(self.core()?.s.bits)
    }

    #[getter]
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Vec<(i64, &'static str)>> {
        let _ = py;
        Ok(self
            .core()?
            .s
            .events
            .iter()
            .map(|&(c, n)| (c, NAMES[n as usize]))
            .collect())
    }

    /// Forget the events collected so far.
    fn clear_events(&self) -> PyResult<()> {
        self.core()?.s.events.clear();
        Ok(())
    }

    #[getter]
    fn ram(slf: &Bound<'_, Self>) -> PyResult<Py<Memory>> {
        Self::view(slf, 0)
    }

    #[getter]
    fn colour(slf: &Bound<'_, Self>) -> PyResult<Py<Memory>> {
        Self::view(slf, 1)
    }

    /// The 16 4 KiB pages as the CPU maps them: RAM, a ROM image, or `None` for I/O.
    #[getter]
    fn pages(slf: &Bound<'_, Self>) -> PyResult<Vec<Option<Py<Memory>>>> {
        let map = slf.get().core()?.s.map;
        map.iter()
            .map(|&k| match k {
                moses_core::sys::IO => Ok(None),
                k => Self::view(slf, if k == 0 { 0 } else { k + 1 }).map(Some),
            })
            .collect()
    }

    /// The REU at `$DF00`, or `None`.
    #[getter]
    fn reu(slf: &Bound<'_, Self>) -> PyResult<Option<Reu>> {
        Ok(slf.get().core()?.s.reu.as_ref().map(|_| Reu {
            machine: slf.clone().unbind(),
        }))
    }

    #[getter]
    fn mpu(slf: &Bound<'_, Self>) -> Registers {
        Registers {
            machine: slf.clone().unbind(),
        }
    }

    #[getter]
    fn vic(slf: &Bound<'_, Self>) -> Chip {
        Chip {
            machine: slf.clone().unbind(),
            which: 0,
        }
    }

    #[getter]
    fn cia1(slf: &Bound<'_, Self>) -> Chip {
        Chip {
            machine: slf.clone().unbind(),
            which: 1,
        }
    }

    #[getter]
    fn cia2(slf: &Bound<'_, Self>) -> Chip {
        Chip {
            machine: slf.clone().unbind(),
            which: 2,
        }
    }

    #[getter]
    fn watch(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.core()?.watch {
            Watch::All => Ok(watch_all(py)?.clone_ref(py).into_any()),
            Watch::Set(s) => {
                let pcs = (0..=0xFFFFu32).filter(|&a| s.contains(a as u16));
                Ok(PyFrozenSet::new(py, pcs)?.into_any().unbind())
            }
        }
    }

    #[setter]
    fn set_watch(&self, watch: &Bound<'_, PyAny>) -> PyResult<()> {
        self.core()?.watch = to_watch(watch)?;
        Ok(())
    }

    /// Call `callback(machine)` before decoding at any pc in `watch` (a set or `WATCH_ALL`).
    #[pyo3(signature = (callback, watch))]
    fn set_trap(&self, callback: Py<PyAny>, watch: &Bound<'_, PyAny>) -> PyResult<()> {
        let core = self.core()?;
        core.watch = to_watch(watch)?;
        core.trap_set = true;
        self.hooks().trap = Some(callback);
        Ok(())
    }

    /// Watch nothing until the CPU first reaches `idle_pc`, then everything.
    fn arm_at(&self, idle_pc: u16) -> PyResult<()> {
        let core = self.core()?;
        core.arm_pc = Some(idle_pc);
        core.armed = false;
        core.watch = Watch::Set(AddrSet::default());
        Ok(())
    }

    /// Call `callback(machine)` after a write to `$00`/`$01` once armed.
    fn on_port_write(&self, callback: Option<Py<PyAny>>) -> PyResult<()> {
        self.core()?.port_set = callback.is_some();
        self.hooks().port = callback;
        Ok(())
    }

    /// Call `callback(machine)` before every decode (slow); `None` removes it.
    fn set_decode_hook(&self, callback: Option<Py<PyAny>>) -> PyResult<()> {
        self.core()?.decode_set = callback.is_some();
        self.hooks().decode = callback;
        Ok(())
    }

    fn __getitem__(&self, addr: i64) -> PyResult<u8> {
        let core = self.core()?;
        let mut host = self.host();
        let v = core.peek(&mut host, (addr & 0xFFFF) as u16);
        host.err.map_or(Ok(v), Err)
    }

    /// A read of the I/O page `$D000-$DFFF`, whatever the banking.
    fn io_read(&self, addr: i64) -> PyResult<u8> {
        let core = self.core()?;
        let mut host = self.host();
        let v = core.s.io_read(&mut host, (addr & 0xFFFF) as u16);
        core.s.flush(&mut host);
        host.err.map_or(Ok(v), Err)
    }

    /// A write to the I/O page `$D000-$DFFF` now, bypassing the pending-write queue.
    fn io_write(&self, addr: i64, value: i64) -> PyResult<()> {
        let core = self.core()?;
        let mut host = self.host();
        core.s
            .io_write(&mut host, (addr & 0xFFFF) as u16, byte(value)?);
        core.s.flush(&mut host);
        host.err.map_or(Ok(()), Err)
    }

    fn __setitem__(slf: &Bound<'_, Self>, addr: i64, value: i64) -> PyResult<()> {
        let me = slf.get();
        let core = me.core()?;
        core.poke((addr & 0xFFFF) as u16, byte(value)?);
        Self::port_hook(slf, core)
    }

    /// Run the CPU and the chips up to cycle `limit`, with the GIL released.
    fn run(slf: &Bound<'_, Self>, limit: i64) -> PyResult<()> {
        let me = slf.get();
        let py = slf.py();
        let _claim = me.owner.claim()?;
        let _sids = me
            .logged
            .iter()
            .map(|sid| sid.get().owner.claim())
            .collect::<PyResult<Vec<_>>>()?;
        let mut host = me.host();
        host.me = Some(slf.clone().unbind());
        let h = me.hooks();
        host.hooks = Hooks {
            trap: h.trap.as_ref().map(|c| c.clone_ref(py)),
            decode: h.decode.as_ref().map(|c| c.clone_ref(py)),
            port: h.port.as_ref().map(|c| c.clone_ref(py)),
        };
        // SAFETY: claimed above; the claim is dropped only after the run returns.
        let core = CorePtr(unsafe { &mut **me.core.get() });
        let err = py.detach(|| {
            let core = core;
            // SAFETY: as above; the pointee outlives this closure.
            unsafe { &mut *core.0 }.run(&mut host, limit);
            host.err.take()
        });
        err.map_or(Ok(()), Err)
    }

    /// Bring the chips to `cycle` (default: the power-on delay) and start the CPU at `entry`.
    #[pyo3(signature = (entry, cycle=None))]
    fn reset(slf: &Bound<'_, Self>, entry: u16, cycle: Option<i64>) -> PyResult<i64> {
        let py = slf.py();
        let me = slf.get();
        let cycle = match cycle {
            Some(c) => c,
            None => py
                .import("vagrant_moses.machine")?
                .getattr("boot_cycle")?
                .call1((me.ntsc,))?
                .extract()?,
        };
        let core = me.core()?;
        let mut host = me.host();
        core.reset(&mut host, entry, cycle);
        if let Some(e) = host.err {
            return Err(e);
        }
        Self::port_hook(slf, core)?;
        Ok(cycle)
    }

    /// Arm the SMC recorder over the lift-time `image` (64 KiB) with the per-opcode `table`
    /// (256 bytes, `vagrant_moses.smc.opcode_table`); a re-arm starts afresh.
    #[pyo3(signature = (image, table, banked=true))]
    fn smc_arm(
        &self,
        image: &Bound<'_, PyAny>,
        table: &Bound<'_, PyAny>,
        banked: bool,
    ) -> PyResult<()> {
        let base = self::image(&octets(image)?)?;
        let table: [u8; 256] = octets(table)?
            .try_into()
            .map_err(|_| PyValueError::new_err("the opcode table is 256 bytes"))?;
        self.core()?.s.rec = Some(Box::new(Recorder::new(base, table, banked)));
        Ok(())
    }

    fn smc_disarm(&self) -> PyResult<()> {
        self.core()?.s.rec = None;
        Ok(())
    }

    #[getter]
    fn smc_armed(&self) -> PyResult<bool> {
        Ok(self.core()?.s.rec.is_some())
    }

    /// End the pre-lift phase: call 0 starts, under the bank bits now.
    fn smc_begin(&self) -> PyResult<()> {
        let core = self.core()?;
        let bits = core.s.bits;
        self.rec(core)?.begin(bits);
        Ok(())
    }

    fn smc_next_call(&self) -> PyResult<()> {
        self.rec(self.core()?)?.next_call();
        Ok(())
    }

    /// Index events by frame of `frame_cycles` from the next decode on.
    fn smc_frame_clock(&self, frame_cycles: i64) -> PyResult<()> {
        if frame_cycles <= 0 {
            return Err(PyValueError::new_err("frame_cycles must be positive"));
        }
        self.rec(self.core()?)?.frame_clock(frame_cycles);
        Ok(())
    }

    #[getter]
    fn smc_call(&self) -> PyResult<i32> {
        Ok(self.rec(self.core()?)?.call)
    }

    /// `machine[addr] = value` as the host's own write: no instruction owns the byte, and a
    /// port write resets the bank-bit writer.
    fn smc_host_write(slf: &Bound<'_, Self>, addr: i64, value: i64) -> PyResult<()> {
        let me = slf.get();
        let core = me.core()?;
        let addr = (addr & 0xFFFF) as u16;
        core.poke(addr, byte(value)?);
        let bits = core.s.bits;
        me.rec(core)?.host(addr, bits);
        Self::port_hook(slf, core)
    }

    /// The first decode after this cycle stops the run and sets `smc_timed_out`.
    #[getter]
    fn smc_deadline(&self) -> PyResult<Option<i64>> {
        let d = self.rec(self.core()?)?.deadline;
        Ok((d != i64::MAX).then_some(d))
    }

    #[setter]
    fn set_smc_deadline(&self, cycle: Option<i64>) -> PyResult<()> {
        let r = self.rec(self.core()?)?;
        (r.deadline, r.timed_out) = (cycle.unwrap_or(i64::MAX), false);
        Ok(())
    }

    #[getter]
    fn smc_timed_out(&self) -> PyResult<bool> {
        Ok(self.rec(self.core()?)?.timed_out)
    }

    /// Attribute the last instruction's writes once a run is over.
    fn smc_finish(&self) -> PyResult<()> {
        let core = self.core()?;
        if let Some(mut r) = core.s.rec.take() {
            r.finish(&core.s);
            core.s.rec = Some(r);
        }
        Ok(())
    }

    /// `(writer, writer opcode, role, timing, mapping, first)` of each distinct event, in
    /// the order first seen.
    #[allow(clippy::type_complexity)]
    fn smc_events(
        &self,
    ) -> PyResult<
        Vec<(
            &'static str,
            Option<u8>,
            &'static str,
            &'static str,
            &'static str,
            i64,
        )>,
    > {
        Ok(self
            .rec(self.core()?)?
            .events
            .iter()
            .map(|e| {
                let (w, op, role, timing, mapping) = e.fields();
                (
                    smc::WRITERS[w as usize],
                    op,
                    smc::ROLES[role as usize],
                    smc::TIMINGS[timing as usize],
                    smc::MAPPINGS[mapping as usize],
                    e.first,
                )
            })
            .collect())
    }

    /// `(count, sample)` of events the axes cannot express; the sample holds the first
    /// `(kind, addr, role, pc)`.
    #[allow(clippy::type_complexity)]
    fn smc_unclassified(
        &self,
    ) -> PyResult<(u64, Vec<(&'static str, u16, Option<&'static str>, u16)>)> {
        let r = self.rec(self.core()?)?;
        let sample = r
            .sample
            .iter()
            .map(|&(k, addr, role, pc)| {
                (
                    smc::UNCLASSIFIED[k as usize],
                    addr,
                    (k == smc::U_UNOWNED).then(|| smc::ROLES[role as usize]),
                    pc,
                )
            })
            .collect();
        Ok((r.unclassified, sample))
    }
}

/// The register log and bus-value model of a SID whose chip lives in Python.
///
/// Writes are logged natively. A read of `$19-$1C` calls `on_read(cycle, reg)` and the byte it
/// returns is the register; every other read is the decaying bus value.
#[pyclass(module = "vagrant_moses._core", name = "NativeSid", frozen)]
pub struct NativeSid {
    shared: Arc<SidShared>,
    on_read: Py<PyAny>,
    owner: Owner,
    model: String,
}

impl NativeSid {
    #[allow(clippy::mut_from_ref)]
    fn state(&self) -> PyResult<&mut moses_core::SidState> {
        self.owner.check()?;
        // SAFETY: GIL held and the owner check refuses a thread but the running machine's.
        Ok(unsafe { self.shared.state() })
    }
}

#[pymethods]
impl NativeSid {
    #[new]
    fn new(model: &str, on_read: Py<PyAny>) -> PyResult<Self> {
        if !matches!(model, "6581" | "8580") {
            return Err(PyValueError::new_err("model must be '6581' or '8580'"));
        }
        Ok(NativeSid {
            shared: Arc::new(SidShared::new(model == "8580")),
            on_read,
            owner: Owner::default(),
            model: model.to_owned(),
        })
    }

    #[getter]
    fn model(&self) -> &str {
        &self.model
    }

    fn write(&self, cycle: i64, reg: i64, value: i64) -> PyResult<()> {
        self.state()?.write(cycle, (reg & 0x1F) as u8, byte(value)?);
        Ok(())
    }

    fn read(&self, py: Python<'_>, cycle: i64, reg: i64) -> PyResult<u8> {
        let reg = (reg & 0x1F) as u8;
        if let Some(v) = self.state()?.peek(cycle, reg) {
            return Ok(v);
        }
        let v = byte(self.on_read.call1(py, (cycle, reg))?.extract(py)?)?;
        self.state()?.settle(cycle, v);
        Ok(v)
    }

    /// Every write since the last drain as `(cycle, reg, value)`.
    fn drain_writes(&self) -> PyResult<Vec<(i64, u8, u8)>> {
        Ok(std::mem::take(&mut self.state()?.writes))
    }

    /// The writes since the last drain that changed their register, as `sidtrace` logs them.
    fn drain_rows(&self) -> PyResult<Vec<(i64, u8, u8)>> {
        Ok(std::mem::take(&mut self.state()?.rows))
    }

    #[getter]
    fn last<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.state()?.last))
    }
}

fn to_watch(watch: &Bound<'_, PyAny>) -> PyResult<Watch> {
    if watch.is_instance_of::<WatchAll>() {
        return Ok(Watch::All);
    }
    let mut set = AddrSet::default();
    for a in watch.try_iter()? {
        set.insert((a?.extract::<i64>()? & 0xFFFF) as u16);
    }
    Ok(Watch::Set(set))
}

/// A zero-copy view of the RAM or the colour RAM (bytearray-like).
#[pyclass(module = "vagrant_moses._core", name = "Memory", frozen)]
pub struct Memory {
    machine: Py<Machine>,
    which: u8,
}

impl Memory {
    fn raw(&self, py: Python<'_>) -> PyResult<(*mut u8, usize)> {
        let core = self.machine.get().core()?;
        let _ = py;
        Ok(match self.which {
            1 => (core.s.colour.as_mut_ptr(), 0x400),
            5 => {
                let reu = core
                    .s
                    .reu
                    .as_mut()
                    .ok_or_else(|| PyRuntimeError::new_err("the machine has no REU"))?;
                (reu.ram.as_mut_ptr(), reu.ram.len())
            }
            k => (
                core.s.img[k.saturating_sub(1) as usize].as_mut_ptr(),
                0x10000,
            ),
        })
    }

    #[allow(clippy::mut_from_ref)]
    fn slice(&self, py: Python<'_>) -> PyResult<&mut [u8]> {
        let (p, n) = self.raw(py)?;
        // SAFETY: the pointee is a boxed array the machine owns and `self` keeps alive.
        Ok(unsafe { std::slice::from_raw_parts_mut(p, n) })
    }
}

#[pymethods]
impl Memory {
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.raw(py)?.1)
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let mem = self.slice(py)?;
        if let Ok(slice) = key.cast::<PySlice>() {
            let ind = slice.indices(mem.len() as isize)?;
            let out: Vec<u8> = (0..ind.slicelength)
                .map(|i| mem[(ind.start + i as isize * ind.step) as usize])
                .collect();
            let bytes = PyBytes::new(py, &out);
            return Ok(py
                .import("builtins")?
                .getattr("bytearray")?
                .call1((bytes,))?
                .unbind());
        }
        let i = index(key.extract()?, mem.len())?;
        Ok(mem[i].into_pyobject(py)?.into_any().unbind())
    }

    fn __setitem__(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mem = self.slice(py)?;
        if let Ok(slice) = key.cast::<PySlice>() {
            let ind = slice.indices(mem.len() as isize)?;
            let data = octets(value)?;
            if ind.slicelength != data.len() {
                return Err(PyValueError::new_err("cannot resize the memory"));
            }
            for (i, b) in data.into_iter().enumerate() {
                mem[(ind.start + i as isize * ind.step) as usize] = b;
            }
            return Ok(());
        }
        let i = index(key.extract()?, mem.len())?;
        mem[i] = byte(value.extract()?)?;
        Ok(())
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let (p, n) = slf.get().raw(slf.py())?;
        // SAFETY: the view keeps `slf`, hence the machine and its boxed memory, alive.
        if unsafe { ffi::PyBuffer_FillInfo(view, slf.as_ptr(), p as *mut c_void, n as _, 0, flags) }
            == -1
        {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }
}

fn index(i: isize, len: usize) -> PyResult<usize> {
    let j = if i < 0 { i + len as isize } else { i };
    usize::try_from(j)
        .ok()
        .filter(|&j| j < len)
        .ok_or_else(|| PyIndexError::new_err("index out of range"))
}

/// The REU (`machine.reu`): its RAM and registers.
#[pyclass(module = "vagrant_moses._core", name = "Reu", frozen)]
pub struct Reu {
    machine: Py<Machine>,
}

impl Reu {
    fn get<T>(&self, f: impl FnOnce(&moses_core::reu::Reu) -> T) -> PyResult<T> {
        let core = self.machine.get().core()?;
        core.s
            .reu
            .as_deref()
            .map(f)
            .ok_or_else(|| PyRuntimeError::new_err("the REU is running a DMA cycle"))
    }
}

#[pymethods]
impl Reu {
    /// The expansion RAM (bytearray-like, zero-copy).
    #[getter]
    fn ram(&self, py: Python<'_>) -> PyResult<Py<Memory>> {
        Machine::view(self.machine.bind(py), 5)
    }

    /// Registers `$00-$0A` as `peek` reads them.
    #[getter]
    fn regs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let regs = self.get(|r| (0..REGS).map(|i| r.peek(i)).collect::<Vec<u8>>())?;
        Ok(PyBytes::new(py, &regs))
    }

    /// Register `reg` without a read's side effects.
    fn peek(&self, reg: u8) -> PyResult<u8> {
        self.get(|r| r.peek(reg))
    }

    #[getter]
    fn size_kb(&self) -> PyResult<u32> {
        self.get(|r| r.size_kb)
    }

    #[getter]
    fn armed(&self) -> PyResult<bool> {
        self.get(|r| r.armed)
    }

    #[getter]
    fn busy(&self) -> PyResult<bool> {
        self.get(|r| r.busy)
    }

    #[getter]
    fn latch(&self) -> PyResult<u8> {
        self.get(|r| r.latch)
    }
}

/// The CPU registers (`machine.mpu`).
#[pyclass(module = "vagrant_moses._core", name = "Registers", frozen)]
pub struct Registers {
    machine: Py<Machine>,
}

macro_rules! registers {
    ($(($get:ident, $set:ident, $field:ident, $ty:ty, $mask:expr)),*) => {
        #[pymethods]
        impl Registers {
            $(
                #[getter]
                fn $get(&self) -> PyResult<$ty> {
                    Ok(self.machine.get().core()?.cpu.$field)
                }

                #[setter]
                fn $set(&self, v: i64) -> PyResult<()> {
                    self.machine.get().core()?.cpu.$field = (v & $mask) as $ty;
                    Ok(())
                }
            )*

            /// Cycles of the instructions executed (`MPU.processorCycles`).
            #[getter]
            #[allow(non_snake_case)]
            fn processorCycles(&self) -> PyResult<u64> {
                Ok(self.machine.get().core()?.cpu.cycles)
            }

            #[setter]
            #[allow(non_snake_case)]
            fn set_processorCycles(&self, v: u64) -> PyResult<()> {
                self.machine.get().core()?.cpu.cycles = v;
                Ok(())
            }

            #[getter]
            fn p(&self) -> PyResult<u8> {
                Ok(self.machine.get().core()?.cpu.p())
            }

            #[setter]
            fn set_p(&self, v: i64) -> PyResult<()> {
                self.machine.get().core()?.cpu.set_p((v & 0xFF) as u8);
                Ok(())
            }
        }
    };
}

registers!(
    (pc, set_pc, pc, u16, 0xFFFF),
    (a, set_a, a, u8, 0xFF),
    (x, set_x, x, u8, 0xFF),
    (y, set_y, y, u8, 0xFF),
    (sp, set_sp, sp, u8, 0xFF)
);

/// Read-only view of a chip's registers and beam/timer state (`machine.vic`, `cia1`, `cia2`).
#[pyclass(module = "vagrant_moses._core", name = "Chip", frozen)]
pub struct Chip {
    machine: Py<Machine>,
    which: u8,
}

#[pymethods]
impl Chip {
    /// The register file as written (CIA: the register shadow).
    #[getter]
    fn regs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let core = self.machine.get().core()?;
        Ok(match self.which {
            0 => PyBytes::new(py, &core.s.vic.regs),
            n => PyBytes::new(py, &core.s.cia[n as usize - 1].regs),
        })
    }

    #[getter]
    fn raster_y(&self) -> PyResult<i32> {
        Ok(self.machine.get().core()?.s.vic.raster_y)
    }

    #[getter]
    fn line_cycle(&self) -> PyResult<i32> {
        Ok(self.machine.get().core()?.s.vic.line_cycle)
    }

    #[getter]
    fn irq_flags(&self) -> PyResult<u8> {
        Ok(self.machine.get().core()?.s.vic.irq_flags)
    }

    #[getter]
    fn irq_mask(&self) -> PyResult<u8> {
        Ok(self.machine.get().core()?.s.vic.irq_mask)
    }

    /// `(counter, latch)` of CIA timer `n` (0 = A, 1 = B).
    fn timer(&self, n: usize) -> PyResult<(u16, u16)> {
        if self.which == 0 || n > 1 {
            return Err(PyValueError::new_err("a CIA has timers 0 and 1"));
        }
        let t = &self.machine.get().core()?.s.cia[self.which as usize - 1].t[n];
        Ok((t.timer, t.latch))
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Machine>()?;
    m.add_class::<Memory>()?;
    m.add_class::<Registers>()?;
    m.add_class::<Reu>()?;
    m.add_class::<Chip>()?;
    m.add_class::<NativeSid>()?;
    m.add_class::<WatchAll>()?;
    m.add("WATCH_ALL", watch_all(m.py())?.clone_ref(m.py()))?;
    m.add("NAMES", PyTuple::new(m.py(), NAMES)?)?;
    Ok(())
}
