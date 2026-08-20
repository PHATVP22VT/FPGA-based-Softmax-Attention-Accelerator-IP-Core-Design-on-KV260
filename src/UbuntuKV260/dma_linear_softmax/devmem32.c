/*
 * devmem32.c  -  Thay the devmem tren KV260 (khong co san)
 *
 * Bien dich:  gcc -O2 -o devmem32 devmem32.c
 * Su dung:
 *   sudo ./devmem32 0xa0000000              (doc 32-bit)
 *   sudo ./devmem32 0xa0000000 0x12345678   (ghi 32-bit roi doc lai)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <phys_addr> [write_val]\n", argv[0]);
        return 1;
    }

    unsigned long phys = strtoul(argv[1], NULL, 0);
    unsigned long page = phys & ~0xFFFUL;
    unsigned off = phys & 0xFFF;

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("/dev/mem"); return 1; }

    volatile void *map = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd, page);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }

    volatile uint32_t *ptr = (volatile uint32_t *)((char *)map + off);

    if (argc >= 3) {
        uint32_t val = strtoul(argv[2], NULL, 0);
        *ptr = val;
        printf("Wrote 0x%08x to 0x%lx\n", val, phys);
    }

    printf("Read  0x%08x from 0x%lx\n", *ptr, phys);

    munmap((void *)map, 0x1000);
    close(fd);
    return 0;
}
