use std::collections::VecDeque;

#[derive(Debug, Default)]
pub struct FrameOrder {
    last_ts: Option<i64>,
    pub out_of_order: u64,
}

impl FrameOrder {
    pub fn pop_latest<T>(&mut self, queue: &mut VecDeque<T>, ts: impl Fn(&T) -> i64) -> Option<T> {
        let mut latest = None;
        while let Some(frame) = queue.pop_front() {
            let t = ts(&frame);
            if self.last_ts.is_some_and(|last| t <= last) {
                self.out_of_order += 1;
                continue;
            }
            self.last_ts = Some(t);
            latest = Some(frame);
        }
        latest
    }

    pub fn pop_front_ordered<T>(
        &mut self,
        queue: &mut VecDeque<T>,
        ts: impl Fn(&T) -> i64,
    ) -> Option<T> {
        while let Some(frame) = queue.pop_front() {
            let t = ts(&frame);
            if self.last_ts.is_some_and(|last| t <= last) {
                self.out_of_order += 1;
                continue;
            }
            self.last_ts = Some(t);
            return Some(frame);
        }
        None
    }

    pub fn reset(&mut self) {
        self.last_ts = None;
    }
}

#[derive(Debug)]
pub struct DeferredRelease {
    delay: usize,
    pending: VecDeque<u64>,
    ready: Vec<u64>,
}

impl DeferredRelease {
    pub fn new(delay: usize) -> Self {
        Self {
            delay,
            pending: VecDeque::new(),
            ready: Vec::new(),
        }
    }

    pub fn frame_dropped(&mut self, id: u64) {
        self.pending.push_back(id);
        while self.pending.len() > self.delay {
            if let Some(old) = self.pending.pop_front() {
                self.ready.push(old);
            }
        }
    }

    pub fn take_ready(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.ready)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_older_than_presented_is_dropped() {
        let mut order = FrameOrder::default();
        let mut q: VecDeque<i64> = [300].into();
        assert_eq!(order.pop_latest(&mut q, |t| *t), Some(300));
        q.extend([200]);
        assert_eq!(order.pop_latest(&mut q, |t| *t), None);
        q.extend([250, 400]);
        assert_eq!(order.pop_latest(&mut q, |t| *t), Some(400));
        assert_eq!(order.out_of_order, 2);
    }

    #[test]
    fn repeated_timestamp_is_dropped() {
        let mut order = FrameOrder::default();
        let mut q: VecDeque<i64> = [100].into();
        assert_eq!(order.pop_latest(&mut q, |t| *t), Some(100));
        q.extend([100, 150]);
        assert_eq!(order.pop_latest(&mut q, |t| *t), Some(150));
        assert_eq!(order.out_of_order, 1);
    }

    #[test]
    fn backlog_collapses_to_newest_frame() {
        let mut order = FrameOrder::default();
        let mut q: VecDeque<i64> = [1, 2, 3].into();
        assert_eq!(order.pop_latest(&mut q, |t| *t), Some(3));
        assert!(q.is_empty());
        assert_eq!(order.out_of_order, 0);
    }

    #[test]
    fn dropped_frame_id_is_held_back_for_frames_in_flight() {
        let mut release = DeferredRelease::new(2);
        release.frame_dropped(1);
        release.frame_dropped(2);
        assert!(release.take_ready().is_empty());
        release.frame_dropped(3);
        assert_eq!(release.take_ready(), vec![1]);
        assert!(release.take_ready().is_empty());
        release.frame_dropped(4);
        release.frame_dropped(5);
        assert_eq!(release.take_ready(), vec![2, 3]);
    }

    #[test]
    fn ordered_pop_returns_every_frame_in_order() {
        let mut order = FrameOrder::default();
        let mut q: VecDeque<i64> = [1, 2, 3].into();
        assert_eq!(order.pop_front_ordered(&mut q, |t| *t), Some(1));
        assert_eq!(order.pop_front_ordered(&mut q, |t| *t), Some(2));
        assert_eq!(order.pop_front_ordered(&mut q, |t| *t), Some(3));
        assert_eq!(order.pop_front_ordered(&mut q, |t| *t), None);
        assert_eq!(order.out_of_order, 0);
    }

    #[test]
    fn ordered_pop_skips_stale_frames_and_keeps_rest() {
        let mut order = FrameOrder::default();
        let mut q: VecDeque<i64> = [10, 5, 10, 20, 30].into();
        assert_eq!(order.pop_front_ordered(&mut q, |t| *t), Some(10));
        assert_eq!(order.pop_front_ordered(&mut q, |t| *t), Some(20));
        assert_eq!(q.len(), 1);
        assert_eq!(order.out_of_order, 2);
    }

    #[test]
    fn ordered_pop_after_backward_seek_needs_reset() {
        let mut order = FrameOrder::default();
        let mut q: VecDeque<i64> = [5_000].into();
        assert_eq!(order.pop_front_ordered(&mut q, |t| *t), Some(5_000));
        q.extend([1_000]);
        assert_eq!(order.pop_front_ordered(&mut q, |t| *t), None);
        order.reset();
        q.extend([1_000, 1_033]);
        assert_eq!(order.pop_front_ordered(&mut q, |t| *t), Some(1_000));
        assert_eq!(order.pop_front_ordered(&mut q, |t| *t), Some(1_033));
    }

    #[test]
    fn frames_after_backward_seek_are_presented_once_reset() {
        let mut order = FrameOrder::default();
        let mut q: VecDeque<i64> = [5_000].into();
        assert_eq!(order.pop_latest(&mut q, |t| *t), Some(5_000));
        q.extend([1_000]);
        assert_eq!(order.pop_latest(&mut q, |t| *t), None);
        order.reset();
        q.extend([1_000, 1_033]);
        assert_eq!(order.pop_latest(&mut q, |t| *t), Some(1_033));
    }
}
