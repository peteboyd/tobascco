/* graph.h */
/* author: Peter Boyd */
/* algorithms to include:
        * a cycle basis
        * an iteration over all possible cycles?
        * automorphism calculations, including reductions
        * symmetry operations based on voltages
*/

#include <stdio.h>
#include <string>
#include <vector>

class Edge {
    std::vector<int> voltage;
    int _f, _t;
    public:
    Edge(int f, int t){ _f = f; _t = t; }
    Edge(int f, int t, std::vector<int> volt) {init_edge(f, t, volt);}
    Edge(): _f(0), _t(0) {}
    ~Edge(){};
    void init_edge(int f, int t, std::vector<int> volt);
    int from(){ return _f; }
    int to(){ return _t; }
};

class Vertex{
    int _i, _degree; 
    std::vector<int> _neighbours;
    public:
    void set_index(int id) { _i = id; }
    int index() { return _i; }
    int degree() { return _degree; }
    Vertex(): _i(0), _degree(0) {}
    Vertex(int id): _degree(0) { _i = id; }
    ~Vertex() {};
    void setNeighbour(int j);
    int getNeighbour(int j);
    std::vector<int>::iterator neighbour_it() { return _neighbours.begin(); }
    std::vector<int>::iterator neighbour_end() { return _neighbours.end(); }
};

class Graph {
private:
    std::string name;
    Edge *edges;
    Vertex *vertices;
    int _size;          //size == number of edges in graph
    int _order;         //order == number of vertices in graph

public:
    void setName(std::string s) { name = s; }
    std::string getName() { return name; }
    Graph(): _size(0), _order(0) {}
    Graph(const std::string s): _size(0), _order(0) { setName(s); }
    ~Graph();
    void init_edges(int size); 
    void init_vertices(int order);
    void set_vertices();
    int size() { return _size; }
    int order() { return _order; }
    void add_edge(int i, int j, std::vector<int> volt);
    int* get_connected_edges(int vertex);
    Vertex get_vertex(int i) { return vertices[i]; }
    void MinimumSpanningTree(std::vector<int>&, std::vector<int>&, int);
};

