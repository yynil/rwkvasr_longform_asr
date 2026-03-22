from rwkvasr.training.progress import start_training_progress, update_training_progress


def test_training_progress_updates_step_and_fields() -> None:
    progress, task_id = start_training_progress(total_steps=10, start_step=2, description="unit-train")
    try:
        update_training_progress(
            progress,
            task_id,
            step=3,
            epoch=1,
            loss=1.2345,
            data_time=0.4,
            step_time=0.6,
            total_elapsed=2.0,
            start_step=2,
        )
        task = progress.tasks[0]
        assert int(task.completed) == 3
        assert task.fields["epoch"] == "1"
        assert task.fields["loss"] == "1.2345"
    finally:
        progress.stop()
