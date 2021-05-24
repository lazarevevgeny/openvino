// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_engines.hpp"
#include "interpreter_engine.hpp"

// Builds a class name for a given backend prefix
// The prefix should come from cmake
// Example: INTERPRETER -> INTERPRETER_Engine
// Example: IE_CPU -> IE_CPU_Engine
#define ENGINE_CLASS_NAME(backend) backend##_Engine
