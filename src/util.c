#include "util.h"

FILE *xfopen(char *path, char *mode) {
  FILE *fp;
  if ( (fp = fopen(path, mode)) == NULL) {
    fprintf(stderr, "Error: failed to open file (%s)\n", path);
    exit(EXIT_FAILURE);
  }

  return fp;
}

void xfclose(FILE *fp) {
  fclose(fp);
}

void *xmalloc(size_t size) {
  void *ret;
  if ((ret = malloc(size)) == NULL) {
    fprintf(stderr, "Error: failed to allocate memory\n");
    exit(EXIT_FAILURE);
  }

  return ret;
}

void *xcalloc(size_t count, size_t size) {
  void *ret;
  if ((ret = calloc(count, size)) == NULL) {
    fprintf(stderr, "Error: failed to allocate memory\n");
    exit(EXIT_FAILURE);
  }

  return ret;
}

void xfree(void *ptr) {
  free(ptr);
}
