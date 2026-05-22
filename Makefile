CC=gcc
CFLAGS=-std=c11 -Wall -Wextra -O2
LDFLAGS=-lm
TARGET=decision_tree_robot
SRC=src/main.c src/decision_tree.c src/dataset.c src/tree_visualization.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET) $(TARGET).exe tree_classification.dot tree_classification.png tree_regression.dot tree_regression.png
