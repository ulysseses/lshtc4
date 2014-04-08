#ifndef __PYX_HAVE__kNN__hinter
#define __PYX_HAVE__kNN__hinter


#ifndef __PYX_HAVE_API__kNN__hinter

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

__PYX_EXTERN_C DL_IMPORT(std::vector<std::vector<DOC> >) *word2doc_func(char *);

#endif /* !__PYX_HAVE_API__kNN__hinter */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC inithinter(void);
#else
PyMODINIT_FUNC PyInit_hinter(void);
#endif

#endif /* !__PYX_HAVE__kNN__hinter */
