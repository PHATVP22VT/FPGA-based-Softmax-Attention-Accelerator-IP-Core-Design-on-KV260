#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

int main(int argc, char *argv[])
{
    unsigned long phys = 0xFD615000;
    unsigned long page = phys & ~0xFFFUL;
    unsigned off = phys & 0xFFF;

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("/dev/mem"); return 1; }

    volatile void *map = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd, page);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }

    volatile uint32_t *ptr = (volatile uint32_t *)((char *)map + off);

    uint32_t val = *ptr;
    printf("Read  0x%08x from 0x%lx\n", val, phys);
    
    /* Change bits [1:0] to 00 (32-bit) */
    if (argc > 1) {
        uint32_t new_val = val & ~0x3; /* clear bits 1:0 */
        *ptr = new_val;
        printf("Wrote 0x%08x to 0x%lx (Set HPM0 to 32-bit)\n", new_val, phys);
    }

    munmap((void *)map, 0x1000);
    close(fd);
    return 0;
}
