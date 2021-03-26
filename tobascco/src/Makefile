CC = g++
CFLAGS = -Wall -g
LDFLAGS = -lm

graph.o : graph.c graph.h
	${CC} ${CFLAGS} -c graph.c

all : graph.o
	${CC} ${CFLAGS} graph.o main.c ${LDFLAGS} -o main

clean:
	rm -f *.o main
