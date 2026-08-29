/* wubuv_info.c — Print WUBV container information */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include "wubuv.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s file.wubv\n", argv[0]);
        return 1;
    }
    
    wubuv_print_info(argv[1]);
    return 0;
}
