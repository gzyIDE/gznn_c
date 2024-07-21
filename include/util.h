#ifndef _UTIL_H_INCLUDED_
#define _UTIL_H_INCLUDED_

#include <stdio.h>
#include <stdlib.h>

FILE *xfopen(char *path, char *mode);
void *xmalloc(size_t size);
void *xcalloc(size_t count, size_t size);
void xfree(void *ptr);


#endif //_UTIL_H_INCLUDED_
