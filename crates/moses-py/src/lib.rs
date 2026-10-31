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

use jennings_core::AddrSet;
use moses_core::lines::NAMES;
use moses_core::machine::{K_ABSX, K_ABSY, K_BRANCH, K_INDY, K_NONE, K_RMW};
use moses_core::{Config, Core, Host, Layout, Watch};
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
    devs: &'a [Py<PyAny>],
    sink: Option<&'a Py<PyAny>>,
    hooks: Hooks,
    me: Option<Py<Machine>>,
    active: &'a AtomicPtr<Core>,
    err: Option<PyErr>,
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
            self.devs[dev]
                .call_method1(py, "read", (cycle, reg))
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
    devs: Vec<Py<PyAny>>,
    sids: Py<PyDict>,
    sink: Option<Py<PyAny>>,
    hooks: UnsafeCell<Hooks>,
    owner: Owner,
    active: AtomicPtr<Core>,
    ntsc: bool,
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
    #[pyo3(signature = (ntsc, sids, cia_new, on_event, layouts, ram, roms, lines, cycles_per_line))]
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
    ) -> PyResult<Self> {
        let mut devs = Vec::new();
        let mut map = Vec::new();
        for (k, v) in sids.iter() {
            map.push((k.extract::<u32>()?, devs.len()));
            devs.push(v.unbind());
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
        };
        Ok(Machine {
            core: UnsafeCell::new(Box::new(Core::new(cfg))),
            devs,
            sids: sids.unbind(),
            sink: on_event,
            hooks: UnsafeCell::new(Hooks::default()),
            owner: Owner::default(),
            active: AtomicPtr::new(ptr::null_mut()),
            ntsc,
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
    fn ram(slf: &Bound<'_, Self>) -> Memory {
        Memory {
            machine: slf.clone().unbind(),
            which: 0,
        }
    }

    #[getter]
    fn colour(slf: &Bound<'_, Self>) -> Memory {
        Memory {
            machine: slf.clone().unbind(),
            which: 1,
        }
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
        Ok(if self.which == 0 {
            (core.s.img[0].as_mut_ptr(), 0x10000)
        } else {
            let _ = py;
            (core.s.colour.as_mut_ptr(), 0x400)
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
    fn __len__(&self) -> usize {
        if self.which == 0 {
            0x10000
        } else {
            0x400
        }
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
    m.add_class::<Chip>()?;
    m.add_class::<WatchAll>()?;
    m.add("WATCH_ALL", watch_all(m.py())?.clone_ref(m.py()))?;
    m.add("NAMES", PyTuple::new(m.py(), NAMES)?)?;
    Ok(())
}
