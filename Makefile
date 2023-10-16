
INCLUDES = -I/include
# LIBS = $(DIR)/lib/liblab1.a
CPPFLAGS = $(INCLUDES)
OBJS = test.o utils.o
CFLAGS = -pthread -Wall $(CPPFLAGS)

all: allKernal

allKernal: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)
# allKernal.o philmain.o $(LIBS)


clean:
	rm -f allKernal $(OBJS)