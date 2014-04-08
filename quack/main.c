#include <Python.h>
#include "caller.h"

int main() {
  Py_Initialize();
  initcaller();
  call_quack();
  Py_Finalize();
  return 0;
}