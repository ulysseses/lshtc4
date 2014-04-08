#ifndef __PYX_HAVE__caller
#define __PYX_HAVE__caller


#ifndef __PYX_HAVE_API__caller

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

__PYX_EXTERN_C DL_IMPORT(void) call_quack(void);

#endif /* !__PYX_HAVE_API__caller */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initcaller(void);
#else
PyMODINIT_FUNC PyInit_caller(void);
#endif

#endif /* !__PYX_HAVE__caller */
