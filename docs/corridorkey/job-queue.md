# Job Queue

The pipeline uses a bounded queue with sentinel-based shutdown to connect its three worker stages. This is implemented in `corridorkey.runtime.queue`.

## BoundedQueue

`BoundedQueue` is a thread-safe FIFO queue with a fixed capacity. When full, producers block. This is backpressure: the pipeline naturally throttles to the speed of the slowest stage without unbounded memory growth.

## Shutdown

Shutdown uses a sentinel object (`STOP`). When a producer is done it calls `put_stop()`. The consumer pulls `STOP`, re-puts it so any other consumer on the same queue also sees it, and exits. This propagates shutdown downstream automatically without shared flags or events.

Always check `item is STOP` before using a value from `get()`.

## Assembly Line

The single-GPU pipeline wires three workers through two queues. The preprocessor pushes onto the input queue. The inference worker pulls from the input queue and pushes onto the output queue. The postwrite worker pulls from the output queue and writes to disk.

For multi-GPU, multiple inference workers share the same input and output queues. The last worker to finish sends `STOP` downstream using a shared atomic counter.

## Queue Depth

`PipelineConfig.input_queue_depth` and `output_queue_depth` control the capacity of each queue. The default of 2 keeps one frame in flight and one buffered per stage. Increasing depth uses more VRAM - each preprocessed frame is approximately 64 MB at 2048 resolution.

## Related

- [Clip State Machine](clip-state.md) - How the processing lock interacts with the queue.
- [Reference - job-queue](reference/job-queue.md) - Full symbol reference.
