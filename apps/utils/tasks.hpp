#pragma once

#include <taskflow/taskflow.hpp>

/** Genertic task submodule.
 *
 * The Taskflow library enables us to compose a complex multi-thread application
 * in a hierarchical manner. It is done by first emplacing the tasks to the
 * intermediate `tf::Taskflow` handle, which is in turn emplaced into the parent
 * handle with the `composed_of()` calls.
 */
class Task {
   public:
    tf::Taskflow taskflow;

    virtual ~Task() = default;

    /** Define the task to compute and cache the tile to disk.
     * @param[in] tile configuration of the tile.
     */
    virtual void emplace() = 0;

    /** Schedule the tasks in a multi-threading environment.
     * @param[in] file_lock HDF5 file lock, to work-around the slow mutex in the HDF5 driver.
     */
    virtual void schedule() = 0;
    inline tf::Taskflow& getTaskflow() { return taskflow; }
};