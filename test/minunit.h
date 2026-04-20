/* Taken from http://www.jera.com/techinfo/jtns/jtn002.html */

/* Simple Macros for testing */
#define mu_assert_less(message, a, b)                                          \
  do {                                                                         \
    if (a > b) {                                                               \
      printf("%s: %1.3e > %1.3e\n", message, a, b);                            \
      return message;                                                          \
    }                                                                          \
  } while (0)
#define mu_assert(message, test)                                               \
  do {                                                                         \
    if (!(test))                                                               \
      return message;                                                          \
  } while (0)
/* Announce the test to stderr before running so a crashing test still
 * leaves a breadcrumb in CI logs (stdout may be block-buffered, stderr
 * is not). */
#define mu_run_test(test)                                                      \
  do {                                                                         \
    fprintf(stderr, "-> %s\n", #test);                                         \
    const char *message = test();                                              \
    tests_run++;                                                               \
    if (message)                                                               \
      return message;                                                          \
  } while (0)
