#ifndef _UTIL_H_INCLUDED_
#define _UTIL_H_INCLUDED_

FILE *xfopen(char *path, char *mode);
void *xmalloc(size_t size);
void xfree(void *ptr);


#endif //_UTIL_H_INCLUDED_
