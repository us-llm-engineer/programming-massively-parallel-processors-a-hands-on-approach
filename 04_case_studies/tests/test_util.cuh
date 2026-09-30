// Minimal test helpers: CHECK counts passes/failures with a message; TEST_SUMMARY prints the totals and returns from main.
#pragma once
#include <stdio.h>
static int g_pass = 0, g_fail = 0;
#define CHECK(cond, ...) do { if (cond) { g_pass++; } else { g_fail++; printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)
#define TEST_SUMMARY(name) do { printf("%-28s %3d checks passed, %d failed\n", name, g_pass, g_fail); return g_fail ? 1 : 0; } while (0)
