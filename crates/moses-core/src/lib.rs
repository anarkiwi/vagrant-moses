//! A cycle-exact C64 over [`jennings_core`]: the 6510 cycle loop, VIC-II, two CIAs, a
//! two-phase scheduler and the banked bus.

pub mod cia;
pub mod lines;
pub mod machine;
pub mod sched;
pub mod sys;
pub mod vic;

pub use machine::{Config, Core, Layout, Watch};
pub use sys::Host;
