#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include "graph.h"


std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems){
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty())
            elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

void readfile(const char* filename, Graph &g){
    std::ifstream f(filename);
    std::string line;
    std::vector<std::string> tok;
    int u;
    while(std::getline(f, line)){
        tok = split(line, ' ');
        if (tok[0] == "id"){
            g.setName(tok[1]);
        }
        else if (tok[0] == "key"){
            u = tok.size();
            int nedge = ( (u - 2) / 5);
            g.init_edges(nedge);
            for (int j=2; j < u; j+=5){
                int v1 = atoi(tok[j].c_str()) - 1;
                int v2 = atoi(tok[j+1].c_str()) - 1;
                int vv[3] = { atoi(tok[j+2].c_str()),
                              atoi(tok[j+3].c_str()),
                              atoi(tok[j+4].c_str()) };
                std::vector<int> volt(&vv[0], &vv[0]+3);
                g.add_edge(v1, v2, volt);
            }

        }

    }
    std::cout<<"Graph topology: "<<g.getName()<<std::endl;
    std::cout<<"Number of Edges: "<<g.size()<<std::endl;
    f.close();
}

int main()
{
    Graph g;
    const char* filename="test.arc";
    readfile(filename, g);
    g.set_vertices();
    std::vector<int> tree_nodes, tree_edges;
    int kk = 0;
    //min spanning tree works, but not sure if the recursive function can 'back out' like the python equivalent.
    g.MinimumSpanningTree(tree_nodes, tree_edges, kk);
    std::cout<<tree_nodes.size()<<std::endl;
    for (int jj=0; (unsigned)jj<tree_nodes.size(); jj++){
        std::cout<<tree_nodes[jj]<<" ";
    }
    std::cout<<std::endl;
    for (int kk=0;(unsigned)kk<tree_edges.size(); kk++){
        std::cout<<tree_edges[kk]<<" ";
    }
    std::cout<<std::endl;
    return 0;
}
