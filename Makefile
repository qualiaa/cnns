SRCDIR:=src

SRCS=$(addprefix $(SRCDIR)/,main.cpp MNIST_loader.cpp Network.cpp)
OBJS=$(SRCS:%.cpp=%.o)
PROGNAME=cnet

DATADIR=data
DATA=$(addprefix $(DATADIR)/, \
	 train-images-idx3-ubyte.dat \
	 train-labels-idx1-ubyte.dat \
	 t10k-images-idx3-ubyte.dat \
	 t10k-labels-idx1-ubyte.dat)

CXXFLAGS=-std=c++14 -Isrc
LDLIBS=-larmadillo

default: $(PROGNAME) | $(DATA)

$(PROGNAME): $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(DATA): %.dat : %.gz | $(DATADIR)
	gzip -d -c "$<" | xxd -e | xxd -r > "$@"

%.gz:
	wget -q --directory-prefix "$(DATADIR)" http://yann.lecun.com/exdb/mnist/$(notdir $@)
	
$(DATADIR):
	mkdir "$@"
