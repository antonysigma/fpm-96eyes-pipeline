#pragma once
#include "Halide.h"

namespace vars {

using Halide::Var;

const Var i{"i"};  //!< Real or complex
const Var x{"x"};  //!< column id of the image
const Var y{"y"};  //!< row id of the image
const Var c{"c"};  //!< RGB color channel
const Var k{"k"};  //!< k-th low-res image

}  // namespace vars