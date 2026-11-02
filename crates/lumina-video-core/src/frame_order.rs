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
}
