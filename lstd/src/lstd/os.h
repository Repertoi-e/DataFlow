#pragma once

#include "types/basic_types.h"

LSTD_BEGIN_NAMESPACE

// @AvoidInclude
template <typename T>
struct array;

// @AvoidInclude
struct string;

///
/// This file includes general functions related to platform specific tasks - implemented in platform files accordingly
///

//
// @TODO: Have a memory heap API for creating new heaps and choosing which one is used for allocations.
// Also by default we should have different heaps for small/medium/large objects to prevent memory fragmentation.
// Right now we rely on the user being thoughtful about memory management and not calling new/delete left and right.
//

// Allocates memory by calling OS functions
[[nodiscard("Leak")]] void *os_allocate_block(s64 size);

// Expands/shrinks a memory block allocated by _os_alloc()_.
// This is NOT realloc. When this fails it returns null instead of allocating a new block and copying the contents of the old one.
// That's why it's not called realloc.
[[nodiscard("Leak")]] void *os_resize_block(void *ptr, s64 newSize);

// Returns the size of a memory block allocated by _os_alloc()_ in bytes
s64 os_get_block_size(void *ptr);

// Frees a memory block allocated by _os_alloc()_
void os_free_block(void *ptr);

// Creates/opens a shared memory block and writes data to it (use this for communication between processes)
void os_write_shared_block(const string &name, void *data, s64 size);

// Read data from a shared memory block (use this for communication between processes)
void os_read_shared_block(const string &name, void *out, s64 size);

// Exits the application with the given exit code.
// Also runs all callbacks registered with _exit_schedule()_.
void os_exit(s32 exitCode = 0);

// Returns a time stamp that can be used for time-interval measurements
time_t os_get_time();

// Converts a time stamp acquired by _os_get_time()_ to seconds
f64 os_time_to_seconds(time_t time);

// Don't free the result of this function. This library follows the convention that if the function is not marked as [[nodiscard]], the returned value should not be freed.
string os_get_clipboard_content();
void os_set_clipboard_content(const string &content);

// Returns the path of the current executable or dynamic library (full dir + name).
//
// Don't free the result of this function. This library follows the convention that if the function is not marked as [[nodiscard]], the returned value should not be freed.
string os_get_current_module();

// Returns the current directory of the current process.
// [Windows] The docs say that SetCurrentDirectory/GetCurrentDirectory
//           are not thread-safe but we use a lock so these are.
//
// Don't free the result of this function. This library follows the convention that if the function is not marked as [[nodiscard]], the returned value should not be freed.
string os_get_working_dir();

// Sets the current directory of the current process (needs to be absolute).
// [Windows] The docs say that SetCurrentDirectory/GetCurrentDirectory
//           are not thread-safe but we use a lock so these are.
void os_set_working_dir(const string &dir);

struct os_get_env_result {
    string Value;
    bool Success;
};

// Get the value of an environment variable, returns true if found.
// If not found and silent is false, logs warning.
// The caller is responsible for freeing the string in the returned value.
[[nodiscard("Leak")]] os_get_env_result os_get_env(const string &name, bool silent = false);

// Sets a variable (creates if it doesn't exist yet) in this process' environment
void os_set_env(const string &name, const string &value);

// Delete a variable from the current process' environment
void os_remove_env(const string &name);

// Get a list of parsed command line arguments excluding the first one.
// Normally the first one is the exe name - you can get that with os_get_current_module().
//
// Don't free the result of this function. This library follows the convention that if the function is not marked as [[nodiscard]], the returned value should not be freed.
array<string> os_get_command_line_arguments();

// Returns an ID which uniquely identifies the current process on the system
u32 os_get_pid();

// Reads input from the console (at most 1 KiB).
// Subsequent calls overwrite an internal buffer, so you need to save the information before that.
//
// Don't free the result of this function. This library follows the convention that if the function is not marked as [[nodiscard]], the returned value should not be freed.
bytes os_read_from_console();

// Utility to report hresult errors produces by calling windows functions.
// Shouldn't be used on other platforms
#if OS == WINDOWS

// Logs a formatted error message.
void windows_report_hresult_error(long hresult, const string &call, const string &file, s32 line);

// CHECKHR checks the return value of _call_ and if the returned HRESULT is less than zero, reports an error.
#define WIN32_CHECKHR(call)                                                              \
    {                                                                                    \
        long result = call;                                                              \
        if (result < 0) windows_report_hresult_error(result, #call, __FILE__, __LINE__); \
    }

// CHECKHR_BOOL checks the return value of _call_ and if the returned is false, reports an error.
#define WIN32_CHECKBOOL(call)                                                                                     \
    {                                                                                                             \
        bool result = call;                                                                                       \
        if (!result) windows_report_hresult_error(HRESULT_FROM_WIN32(GetLastError()), #call, __FILE__, __LINE__); \
    }

// DX_CHECK is used for checking directx calls. The difference from WIN32_CHECKHR is that
// in Dist configuration, the macro expands to just the call (no error checking).
#if defined DEBUG || defined RELEASE
#define DX_CHECK(call) WIN32_CHECKHR(call)
#else
#define DX_CHECK(call) call
#endif

#define COM_SAFE_RELEASE(x) \
    if (x) {                \
        x->Release();       \
        x = null;           \
    }

#endif  // OS == WINDOWS

LSTD_END_NAMESPACE
