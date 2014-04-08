#include "graph.h"
#include <vector>
#include <iostream>
#include <algorithm> //std::find
#include <iterator> //std::begin, std::end

void Graph::init_edges(int size) {
    edges = new Edge[size];
    _size = size;
}

void Graph::init_vertices(int order) {
    vertices = new Vertex[order];
    for (int j = 0; j < order; j++){
        vertices[j].set_index(j);
    }
    _order = order;
}

void Graph::add_edge(int i, int j, std::vector<int> volt){
    edges[_order].init_edge(i, j, volt);
    _order ++;
}

void Graph::set_vertices(){
    //assuming all edges have been read in
    int vi, vj;
    int vmax = 0;
    if (_size == 0) return;
    for (int i = 0; i < _size; i++){
        //init screen for max vertices
        if(edges[i].from() > vmax) {vmax = edges[i].from();}
        if(edges[i].to() > vmax) {vmax = edges[i].to();}
    }
    init_vertices(vmax+1); //+1 to account for the 0th index
    for (int i = 0; i < _size; i++){ 
        //add neighbour info
        vi = edges[i].from();
        vj = edges[i].to();
        vertices[vi].setNeighbour(vj);
        vertices[vj].setNeighbour(vi);
    }
}

void Edge::init_edge(int f, int t, std::vector<int> volt){
    _f = f;
    _t = t;
    voltage = volt;
}

void Vertex::setNeighbour(int id){
    bool push = true;
    for (int i=0; i<_degree; i++){
        if (_neighbours[i] == id)
            push = false;
    }
    if (push)
    {
        _neighbours.push_back(id);
        _degree ++;
    }
}

int Vertex::getNeighbour(int id){
    return _neighbours[id];
}

//return array of edges which are incident on 
// this vertex. negative sign placed on edges which
// point "in" to the vertex
int* Graph::get_connected_edges(int v){
    int size = vertices[v].degree(), ind = 0;
    int* edge_container = new int[size];
    for (int ii = 0; ii < _size; ii++){
        if (edges[ii].from() == v)
        {
            edge_container[ind] = ii;
            ind++;
        }
        else if (edges[ii].to() == v)
        {
            edge_container[ind] = -ii;
            ind++;
        }
    }
    return edge_container;
}


//Prim's Minimum Spanning Tree algorithm
void Graph::MinimumSpanningTree(std::vector<int> pool, std::vector<int> used, int i){
    std::cout<<"Vertex: "<<i<<std::endl;
    const int nsize = vertices[i].degree();
    //get neighbours
    int* nn;
    nn = get_connected_edges(i);
    for (int jj = 0; jj < nsize; jj++){
        int ind = nn[jj], newv;
        std::cout<<pool.size()<<std::endl;
        if (ind < 0)
            newv = edges[std::abs(ind)].from();
        else
            newv = edges[ind].to();
        bool exists = std::find(pool.begin(), pool.end(), newv) != pool.end();
        //bool exists = false;
        if (!exists){
            pool.push_back(i);
            used.push_back(nn[jj]);
            MinimumSpanningTree(pool, used, newv);
        }
    /*for (int j=0; j < _size; j++)
    {
        
        std::cout<<"Vertex: "<<j<<std::endl;
        for (int k=0; k < vertices[j].degree(); k++)
            std::cout<<vertices[j].getNeighbour(k)<<" ";
        std::cout<<std::endl;
    }
    */
    }
}

Graph::~Graph(){
    if (vertices) delete [] vertices;
    if (edges) delete [] edges;
    
}
