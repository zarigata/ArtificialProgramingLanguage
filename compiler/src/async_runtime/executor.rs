// Async Executor
// Runs async tasks to completion

use super::*;
use std::collections::VecDeque;
use std::pin::Pin;
use std::thread;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::marker::Unpin;

// Task executor
pub struct Executor {
    tasks: VecDeque<Box<dyn Task>>,
    ready_queue: VecDeque<TaskId>,
}

pub type TaskId = usize;

pub trait Task {
    fn poll(&mut self, waker: &Waker) -> Poll<()>;
    fn id(&self) -> TaskId;
}

impl Executor {
    pub fn new() -> Self {
        Executor {
            tasks: VecDeque::new(),
            ready_queue: VecDeque::new(),
        }
    }
    
    pub fn spawn<F>(&mut self, future: F) -> TaskId
    where
        F: Future<Output = ()> + 'static,
    {
        let task_id = self.tasks.len();
        let task = TaskWrapper::new(task_id, future);
        self.tasks.push_back(Box::new(task));
        self.ready_queue.push_back(task_id);
        task_id
    }
    
    pub fn run(&mut self) {
        while let Some(task_id) = self.ready_queue.pop_front() {
            if let Some(task) = self.tasks.get_mut(task_id) {
                let waker = Waker::new(wake_task, task_id as *const ());
                
                match task.poll(&waker) {
                    Poll::Ready(()) => {
                        // Task completed, remove it
                        // In production, we'd handle this more efficiently
                    }
                    Poll::Pending => {
                        // Task not ready, will be woken later
                    }
                }
            }
        }
    }
    
    pub fn block_on<F>(&mut self, mut future: F) -> F::Output
    where
        F: Future + Unpin,
    {
        let waker = Waker::new(|_| {}, std::ptr::null());
        
        loop {
            match Pin::new(&mut future).poll(&waker) {
                Poll::Ready(value) => return value,
                Poll::Pending => {
                    // Run other tasks
                    self.run();
                }
            }
        }
    }
}

// Task wrapper
struct TaskWrapper<F> {
    id: TaskId,
    future: F,
}

impl<F> TaskWrapper<F>
where
    F: Future<Output = ()>,
{
    fn new(id: TaskId, future: F) -> Self {
        TaskWrapper { id, future }
    }
}

impl<F> Task for TaskWrapper<F>
where
    F: Future<Output = ()>,
{
    fn poll(&mut self, waker: &Waker) -> Poll<()> {
        self.future.poll(waker)
    }
    
    fn id(&self) -> TaskId {
        self.id
    }
}

// Wake function for tasks
fn wake_task(data: *const ()) {
    let _task_id = data as TaskId;
    // In production, this would notify the executor
    // that task_id is ready to run
}

// Thread pool executor
pub struct ThreadPoolExecutor {
    threads: Vec<thread::JoinHandle<()>>,
    task_queue: Arc<Mutex<VecDeque<Box<dyn Task + Send>>>>,
}

impl ThreadPoolExecutor {
    pub fn new(num_threads: usize) -> Self {
        let task_queue: Arc<Mutex<VecDeque<Box<dyn Task + Send>>>> = Arc::new(Mutex::new(VecDeque::new()));
        let mut threads = Vec::new();
        
        for _ in 0..num_threads {
            let queue = task_queue.clone();
            let thread = thread::spawn(move || {
                loop {
                    let task = {
                        let mut queue = queue.lock().unwrap();
                        queue.pop_front()
                    };
                    
                    if let Some(mut task) = task {
                        let waker = Waker::new(|_| {}, std::ptr::null());
                        task.as_mut().poll(&waker);
                    } else {
                        // No tasks, sleep briefly
                        thread::sleep(std::time::Duration::from_millis(10));
                    }
                }
            });
            threads.push(thread);
        }
        
        ThreadPoolExecutor {
            threads,
            task_queue,
        }
    }
    
    pub fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let task = TaskWrapper::new(0, future);
        let mut queue = self.task_queue.lock().unwrap();
        queue.push_back(Box::new(task));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_executor() {
        let mut executor = Executor::new();
        
        struct TestFuture {
            count: i32,
        }
        
        impl Future for TestFuture {
            type Output = ();
            
            fn poll(&mut self, _waker: &Waker) -> Poll<Self::Output> {
                self.count += 1;
                if self.count >= 3 {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            }
        }
        
        executor.spawn(TestFuture { count: 0 });
        executor.run();
    }
}
