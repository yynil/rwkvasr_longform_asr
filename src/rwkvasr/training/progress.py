from __future__ import annotations

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def start_training_progress(
    *,
    total_steps: int,
    start_step: int = 0,
    description: str = "train",
) -> tuple[Progress, TaskID]:
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("epoch={task.fields[epoch]}"),
        TextColumn("loss={task.fields[loss]}"),
        TextColumn("data={task.fields[data_time]}"),
        TextColumn("step={task.fields[step_time]}"),
        TextColumn("rate={task.fields[rate]}"),
        transient=False,
        refresh_per_second=2,
    )
    progress.start()
    task_id = progress.add_task(
        description,
        total=total_steps,
        completed=start_step,
        epoch="0",
        loss="n/a",
        data_time="n/a",
        step_time="n/a",
        rate="n/a",
    )
    return progress, task_id


def update_training_progress(
    progress: Progress,
    task_id: TaskID,
    *,
    step: int,
    epoch: int,
    loss: float,
    data_time: float,
    step_time: float,
    total_elapsed: float,
    start_step: int = 0,
) -> None:
    elapsed_steps = max(step - start_step, 1)
    rate = elapsed_steps / max(total_elapsed, 1.0e-6)
    progress.update(
        task_id,
        completed=step,
        epoch=str(epoch),
        loss=f"{loss:.4f}",
        data_time=f"{data_time:.2f}s",
        step_time=f"{step_time:.2f}s",
        rate=f"{rate:.2f} step/s",
    )
