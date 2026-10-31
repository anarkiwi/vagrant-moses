//! A two-phase discrete event scheduler with libsidplayfp's ordering.
//!
//! Time counts half-cycles: cycle `t` has a PHI1 slot `2t` (VIC-II, CIAs) and a
//! PHI2 slot `2t + 1` (CPU). Events due at the same slot fire in scheduling order.

pub const PHI1: i64 = 0;
pub const PHI2: i64 = 1;
pub const NEVER: i64 = 1 << 62;

/// Event identifiers: the VIC-II and the machine's IRQ clear, then nine per CIA.
pub const E_VIC: u8 = 0;
pub const E_BADLINE: u8 = 1;
pub const E_EDGE: u8 = 2;
pub const E_CLEAR_INT: u8 = 3;
pub const E_CIA: u8 = 4;
pub const CIA_EVENTS: u8 = 9;
pub const N_EVENTS: usize = 4 + 2 * CIA_EVENTS as usize;

/// Offsets of a CIA's events from its base.
pub const C_TIMER: u8 = 0;
pub const C_SKIP: u8 = 1;
pub const C_INTERRUPT: u8 = 4;
pub const C_UPDATE_IDR: u8 = 5;
pub const C_SET_IRQ: u8 = 6;
pub const C_CLEAR_IRQ: u8 = 7;
pub const C_B_TICK: u8 = 8;

/// Pending events in trigger order; `now` is the current half-cycle.
#[derive(Clone)]
pub struct Sched {
    pub now: i64,
    pub next: i64,
    trigger: [i64; N_EVENTS],
    pending: [bool; N_EVENTS],
    queue: [u8; N_EVENTS],
    len: usize,
}

impl Default for Sched {
    fn default() -> Self {
        Self::new()
    }
}

impl Sched {
    pub fn new() -> Self {
        Sched {
            now: 0,
            next: NEVER,
            trigger: [NEVER; N_EVENTS],
            pending: [false; N_EVENTS],
            queue: [0; N_EVENTS],
            len: 0,
        }
    }

    fn remove(&mut self, ev: u8) {
        let i = self.queue[..self.len]
            .iter()
            .position(|&e| e == ev)
            .expect("pending event is queued");
        self.queue.copy_within(i + 1..self.len, i);
        self.len -= 1;
    }

    fn refresh(&mut self) {
        self.next = if self.len == 0 {
            NEVER
        } else {
            self.trigger[self.queue[0] as usize]
        };
    }

    fn insert(&mut self, ev: u8, trigger: i64) {
        if self.pending[ev as usize] {
            self.remove(ev);
        }
        self.trigger[ev as usize] = trigger;
        self.pending[ev as usize] = true;
        let mut i = self.len;
        while i > 0 && self.trigger[self.queue[i - 1] as usize] > trigger {
            i -= 1;
        }
        self.queue.copy_within(i..self.len, i + 1);
        self.queue[i] = ev;
        self.len += 1;
        self.refresh();
    }

    /// Fire `ev` `cycles` cycles after the next `phase` slot from now.
    pub fn schedule(&mut self, ev: u8, cycles: i64, phase: i64) {
        let now = self.now;
        self.insert(ev, now + ((now & 1) ^ phase) + (cycles << 1));
    }

    /// Fire `ev` `cycles` whole cycles from now, in the current phase.
    pub fn schedule_same_phase(&mut self, ev: u8, cycles: i64) {
        self.insert(ev, self.now + (cycles << 1));
    }

    /// Drop `ev` if it is pending.
    pub fn cancel(&mut self, ev: u8) {
        if self.pending[ev as usize] {
            self.pending[ev as usize] = false;
            self.remove(ev);
            self.refresh();
        }
    }

    pub fn is_pending(&self, ev: u8) -> bool {
        self.pending[ev as usize]
    }

    /// The current cycle as seen from `phase` (libsidplayfp `getTime`).
    #[inline(always)]
    pub fn time(&self, phase: i64) -> i64 {
        (self.now + (phase ^ 1)) >> 1
    }

    /// Pop the earliest event due at or before `limit`, advancing `now` to it.
    #[inline]
    pub fn pop_due(&mut self, limit: i64) -> Option<u8> {
        if self.next > limit {
            return None;
        }
        let ev = self.queue[0];
        self.queue.copy_within(1..self.len, 0);
        self.len -= 1;
        self.refresh();
        self.pending[ev as usize] = false;
        self.now = self.trigger[ev as usize];
        Some(ev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_trigger_fires_in_scheduling_order() {
        let mut s = Sched::new();
        s.schedule(5, 1, PHI1);
        s.schedule(2, 1, PHI1);
        s.schedule(7, 0, PHI1);
        let order: Vec<u8> = std::iter::from_fn(|| s.pop_due(10)).collect();
        assert_eq!(order, [7, 5, 2]);
    }

    #[test]
    fn reschedule_moves_and_cancel_drops() {
        let mut s = Sched::new();
        s.schedule(1, 4, PHI1);
        s.schedule(2, 1, PHI1);
        s.schedule(1, 0, PHI1);
        s.cancel(2);
        assert!(!s.is_pending(2));
        assert_eq!(s.pop_due(0), Some(1));
        assert_eq!(s.pop_due(100), None);
        assert_eq!(s.next, NEVER);
    }

    #[test]
    fn phases_and_time() {
        let mut s = Sched::new();
        s.now = 7;
        s.schedule(0, 0, PHI1);
        assert_eq!(s.next, 8);
        s.schedule(0, 1, PHI2);
        assert_eq!(s.next, 9);
        assert_eq!(s.time(PHI1), 4);
        assert_eq!(s.time(PHI2), 3);
    }
}
