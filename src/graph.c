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
int * Graph::get_connected_edges(int v){
    int size = vertices[v].degree(), ind = 0;
    int *edge_container = new int[size];
    for (int ii = 0; ii < _size; ii++){
        if ((edges[ii].from() == v) || (edges[ii].to() == v))
        {
            edge_container[ind] = ii;
            ind++;
        }
    }
    return edge_container;
}


//Prim's Minimum Spanning Tree algorithm
void Graph::MinimumSpanningTree(std::vector<int>& pool, std::vector<int>& used, int i){
    //std::cout<<"Vertex: "<<i<<std::endl;
    const int nsize = vertices[i].degree();
    //get neighbours
    if (pool.size() != (unsigned)_order){
        pool.push_back(i);
        //std::cout<<pool.size()<<" "<<_order<<std::endl;
        int* nn;
        nn = get_connected_edges(i);
        /*
        std::cout<<"Vertex "<<i<<" Edges: ";
        for (int kkk = 0; kkk < nsize; kkk++){
            std::cout<<nn[kkk]<<" ";
        }
        std::cout<<std::endl;
        */
        for (int jj = 0; jj < nsize; jj++){
            int ind = nn[jj], newv;
            //std::cout<<pool.size()<<std::endl;
            Edge e = edges[ind];
            if (e.from() == i)
                newv = e.to();
            else
                newv = e.from();
            bool node_exists = std::find(pool.begin(), pool.end(), newv) != pool.end();
            bool edge_exists = std::find(used.begin(), used.end(), ind) != used.end();
            if (!node_exists && !edge_exists){ 
                used.push_back(ind);
                MinimumSpanningTree(pool, used, newv);
                //used.pop_back();
            }
        }
        delete [] nn;
        //pool.pop_back();
    }
}

Graph::~Graph(){
    if (vertices) delete [] vertices;
    if (edges) delete [] edges;
}
