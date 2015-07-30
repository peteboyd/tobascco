#!/usr/bin/env sage-python 
import logging
import sys
from logging import info, debug, warning, error, critical
import config
import pickle
from config import Terminate
import glog
from time import time
import ConfigParser
from Generator import Generate
from Visualizer import GraphPlot 
from CSV import CSV
from Net import SystreDB, Net
from Builder import Build
from SecondaryBuildingUnit import SBU
from CreateInput import SBUFileRead
from random import randint
import itertools
import numpy as np
import os

try:
    __version_info__ = (0, 0, 0, int("$Revision$".strip("$Revision: ")))
except ValueError:
    __version_info__ = (0, 0, 0, 0)
__version__ = "%i.%i.%i.%i"%__version_info__

class JobHandler(object):
    """determines what job(s) to run based on arguments from the
    options class.
    
    """
    def __init__(self, options):
        self.options = options
        self._topologies = SystreDB()
        self._stored_nets = {}
        self.sbu_pool = []

    def direct_job(self):
        """Reads the options and decides what to do next."""
        
        # TODO(pboyd): problem reading in openbabel libraries for the inputfile
        #  creation due to the use of a custom python implemented for sage.


        if self.options.create_sbu_input_files:
            info("Creating input files")
            job = SBUFileRead(self.options)
            job.read_sbu_files()
            job.sort_sbus()
            job.write_file()
            Terminate()

        self._read_sbu_database_files()
        self._read_topology_database_files()
        # failsafe in case no topology is requested in the input file.
        if not self.options.topologies:
            self.options.topologies = self._topologies.keys()
            debug("No topologies requested, trying all of the ones in the SBU database files." + 
                  " These are %s"%", ".join(self.options.topologies))
            
        # failsafe in case no organic sbus requested in the input file.
        if not self.options.organic_sbus:
            self.options.organic_sbus = [sbu.identifier for sbu in self.sbu_pool 
                                         if not sbu.is_metal]
        # failsafe in case no metal sbus requested in the input file.
        if not self.options.metal_sbus:
            self.options.metal_sbus = [sbu.identifier for sbu in self.sbu_pool 
                                       if sbu.is_metal]
        
        if self.options.calc_sbu_surface_area or self.options.calc_max_sbu_span:
            info("SBU report requested..")
            self._pop_unwanted_sbus()
            self._sbu_report()
            # Currently terminates without trying to build if a report on the 
            # sbu data is requested.. this can be changed.
            Terminate()
        self._pop_unwanted_sbus()
        self._pop_unwanted_topologies()
        self._build_structures()
        #self._build_structures_from_top()

    def _check_barycentric_embedding(self, graph, voltage):
        net = Net(graph)
        net.voltage = voltage
        net.simple_cycle_basis()
        net.get_lattice_basis()
        #net.get_cycle_basis()
        net.get_cocycle_basis()
        #for i,j in itertools.combinations(range(len(net.kernel)), 2):
        #    print np.any(np.in1d(np.array(net.kernel)[i].nonzero(), np.array(net.kernel)[j].nonzero()))

        #for i, j in itertools.combinations(range(len(net.cycle)), 2):
        #    if not np.any(np.in1d(np.array(net.cycle)[i].nonzero(), np.array(net.kernel)[j].nonzero())):
        #        print 'i', ', '.join(['e%i'%(k+1) for k in np.nonzero(np.array(net.kernel)[i])[0]])
        #        print 'j', ', '.join(['e%i'%(k+1) for k in np.nonzero(np.array(net.kernel)[j])[0]])
        #print np.array(net.cycle)[0].nonzero(), np.array(net.cycle)[1].nonzero()
        net.barycentric_embedding()
        #verts = net.graph.vertices()
        #for id in range(len(verts)):
        #    Pi = [verts[id], verts[:id] + verts[id+1:]]
        #    print net.graph.to_undirected().coarsest_equitable_refinement(Pi)
        #G = net.graph.to_undirected().dominating_set(independent=True)
        #for i in np.array(net.cycle):
        #    print ', '.join(['e%i'%(k+1) for k in np.nonzero(i)[0]])

        #q = np.concatenate((net.cycle, net.kernel[:8]))
        #for id, volt in enumerate(np.array(net.voltage)):
        #    print 'e%i'%(id+1), "(%i, %i, %i)"%(tuple(volt))

        #A = matrix(q)
        #for i in A.echelon_form():
        #    print ', '.join(['e%i'%(k+1) for k in np.nonzero(i)[0]])
        #for j in np.array(net.kernel):
        #    print ', '.join(['e%i'%(k+1) for k in np.nonzero(j)[0]])

        #print G.order()
        #print G.gens()
        g = GraphPlot(net)
        #g.view_graph()
        #g.view_placement(init=(0.625, 0.625, 0.625))
        #g.view_placement(init=(0.5, 0.5, 0.5), edge_labels=False, sbu_only=["1"]) # for bcu for paper
        g.view_placement(init=(0.5, 0.5, 0.5), edge_labels=False) # for bcu for paper


    def _build_structures_from_top(self):
        if not self._topologies:
            warning("No topologies found!")
            Terminate()

        csvinfo = CSV(name='%s_info'%(self.options.jobname))
        csvinfo.set_headings('topology', 'sbus', 'edge_count', 'time', 'space_group')
        csvinfo.set_headings('edge_length_err', 'edge_length_std', 'edge_angle_err', 'edge_angle_std')
        self.options.csv = csvinfo
        run = Generate(self.options, self.sbu_pool)
        inittime = time()
        for top, graph in self._topologies.items():
            if self.options.show_barycentric_net_only:
                info("Preparing barycentric embedding of %s"%(top))
                self._check_barycentric_embedding(graph, self._topologies.voltages[top])
            else:

                build = Build(self.options)
                build.net = (top, graph, self._topologies.voltages[top])
                if self.options.sbu_combinations:
                    combinations = run.combinations_from_options()
                else:
                    combinations = run.generate_sbu_combinations(incidence=build.net_degrees())

                if not list(combinations):
                    debug("Net %s does not support the same"%(top)+
                            " connectivity offered by the SBUs")
                for combo in combinations:
                    build.sbus = list(set(combo))
                    # check node incidence
                    if build.met_met_bonds and run.linear_sbus_exist:
                        # add linear organics
                        debug("Metal-type nodes attached to metal-type nodes. "+
                                "Attempting to insert 2-c organic SBUs between these nodes.")
                        for comb in run.yield_linear_org_sbu(combo):
                            build.sbus = list(set(comb))
                            self.embed_sbu_combo(top, comb, build)
                    elif build.met_met_bonds and not run.linear_sbus_exist:
                        debug("Metal-type nodes are attached to metal-type nodes. "+
                                "No linear SBUs exist in database, so the structure "+
                                "will have metal - metal SBUs joined")
                        self.embed_sbu_combo(top, combo, build)
                    else:
                        self.embed_sbu_combo(top, combo, build)


        finaltime = time() - inittime
        info("Topcryst completed after %f seconds"%finaltime)
        Terminate()

    def combo_str(self, combo):
        str = "("
        for j in set(combo):
            str += "%s, "%j.name
        return str[:-2]+")"

    def embed_sbu_combo(self, top, combo, build):
        count = build.net.original_graph.size()
        self.options.csv.add_data(**{"topology.1":top, 
            "sbus.1":self.combo_str(combo),
            "edge_count.1":count})
        info("Setting up %s"%(self.combo_str(combo)) +
                " with net %s, with an edge count = %i "%(top, count))
        t1 = time()
        build.init_embed()
        debug("Augmented graph consists of %i vertices and %i edges"%
                (build.net.order, build.net.shape))
        build.assign_vertices()
        build.assign_edges()
        build.obtain_embedding()
        t2 = time()
        if build.success:
            sym = build.struct.space_group_name
            self.options.csv.add_data(**{"net_charge.1":build.struct.charge})
            if self.options.store_net:
                self._stored_nets[build.name] = build.embedded_net
        else:
            sym = "None"
        self.options.csv.add_data(**{"time.1":t2-t1,
                                    "space_group.1":sym})
        
        #build.custom_embedding(rep, mt)
        if self.options.show_embedded_net:
            build.show()

    def _build_structures(self):
        """Pass the sbu combinations to a MOF building algorithm."""
        run = Generate(self.options, self.sbu_pool)
        # generate the combinations of SBUs to build
        if self.options.sbu_combinations:
            combinations = run.combinations_from_options()
        else:
            # remove SBUs if not listed in options.organic_sbus or options.metal_sbus
            combinations = run.generate_sbu_combinations()
        csvinfo = CSV(name='%s_info'%(self.options.jobname))
        csvinfo.set_headings('topology', 'sbus', 'edge_count', 'time', 'space_group', 'net_charge')
        csvinfo.set_headings('edge_length_err', 'edge_length_std', 'edge_angle_err', 'edge_angle_std')
        self.options.csv = csvinfo
        # generate the MOFs.
        inittime = time()
        for combo in combinations:
            node_degree = [i.degree for i in set(combo)]
            node_lin = [i.linear for i in set(combo)]
            degree = sorted([j for i, j in zip(node_lin, node_degree) if not i])
            # find degrees of the sbus in the combo
            if not self._topologies:
                warning("No topologies found! Exiting.")
                Terminate()
            debug("Trying "+self.combo_str(combo))
            for top, graph in self._topologies.items():
                build = Build(self.options)
                build.sbus = list(set(combo))
                build.net = (top, graph, self._topologies.voltages[top])
                #build.get_automorphisms()
                if self.options.show_barycentric_net_only:
                    info("Preparing barycentric embedding of %s"%(top))
                    self._check_barycentric_embedding(graph, self._topologies.voltages[top])
                else:
                    if build.check_net:
                        # check node incidence
                        if build.met_met_bonds and run.linear_sbus_exist and not run.linear_in_combo(combo):
                            # add linear organics
                            debug("Metal-type nodes attached to metal-type nodes. "+
                                    "Attempting to insert linear organic SBUs between these nodes.")
                            for comb in run.yield_linear_org_sbu(combo):
                                build = Build(self.options)
                                build.sbus = list(set(comb))
                                build.net = (top, graph, self._topologies.voltages[top])
                                self.embed_sbu_combo(top, comb, build)
                        elif build.met_met_bonds and run.linear_in_combo(combo):
                            self.embed_sbu_combo(top, combo, build)

                        elif build.met_met_bonds and not run.linear_sbus_exist:
                            debug("Metal-type nodes are attached to metal-type nodes. "+
                                   "No linear SBUs exist in database, so the structure "+
                                    "will have metal - metal SBUs joined")
                            self.embed_sbu_combo(top, combo, build)
                        elif not build.met_met_bonds:
                            self.embed_sbu_combo(top, combo, build)
                    else:
                        debug("Net %s does not support the same"%(top)+
                                " connectivity offered by the SBUs")

        finaltime = time() - inittime
        info("Topcryst completed after %f seconds"%finaltime)
        if self.options.get_run_info:
            info("Writing run information to %s"%self.options.csv.filename)
            self.options.csv.write()
        if self.options.store_net and self._stored_nets:
            info("Writing all nets to nets_%s.pkl"%self.options.jobname)
            f = open("nets_%s.pkl"%self.options.jobname, 'wb')
            p = pickle.dump(self._stored_nets, f)
            f.close()
        Terminate()

    def _sbu_report(self):
        """Compute the surface areas and report them to a .csv file."""
        # WARNING - this assumes that SBUs with the same name but in
        # different topologies are the same, and will take the last instance
        
        met_sbus = {}
        org_sbus = {}
        for sbu in self.sbu_pool:
            if sbu.is_metal:
                met_sbus[sbu.name] = sbu
            else:
                org_sbus[sbu.name] = sbu
        filename = os.path.join(self.options.job_dir,
                                self.options.jobname + ".SBU_report.csv")
        report = CSV(name=filename)
        report.set_headings("sbu_id")
        if self.options.calc_sbu_surface_area:
            report.set_headings("surface_area")
        if self.options.calc_max_sbu_span:
            report.set_headings("sbu_span")
        # metal sbus first.
        for name, sbu in met_sbus.items():
            info("Computing data for %s"%name)
            report.add_data(**{"sbu_id.1": sbu.identifier})
            if self.options.calc_sbu_surface_area:
                report.add_data(**{"surface_area.1": sbu.surface_area})
            if self.options.calc_max_sbu_span:
                report.add_data(**{"sbu_span.1":sbu.max_span})
        
        # list organic SBUs second.
        for name, sbu in org_sbus.items():
            info("Computing data for %s"%name)
            report.add_data(**{"sbu_id.1": sbu.identifier})
            if self.options.calc_sbu_surface_area:
                report.add_data(**{"surface_area.1": sbu.surface_area})
            if self.options.calc_max_sbu_span:
                report.add_data(**{"sbu_span.1": sbu.max_span})
        report.write()
         
    def _read_topology_database_files(self):
        for file in self.options.topology_files:
            db = SystreDB(filename=file)
            for top in db.keys():
                if top in self._topologies.keys():
                    warning("Duplicate topologies found! The topology %s"%(top)+
                             " will be represented from the file %s"%(file))
            self._topologies.update(db)
            self._topologies.voltages.update(db.voltages)

    def _read_sbu_database_files(self):
        """Read in the files containing SBUs. Currently supports only the special
        Config .ini file types, but should be easily expandable to different input
        files."""
        
        for file in self.options.sbu_files:
            debug("reading %s"%(file))
            self._from_config(file)
                
    def _from_config(self, filename):
        sbu_config = ConfigParser.SafeConfigParser()
        sbu_config.read(filename)
        info("Found %i SBUs"%(len(sbu_config.sections())))
        for raw_sbu in sbu_config.sections():
            debug("Reading %s"%(raw_sbu))
            sbu = SBU()
            sbu.from_config(raw_sbu, sbu_config)
            self.sbu_pool.append(sbu)
        
        rem = []
        for id, sbu in enumerate(self.sbu_pool):
            if sbu.parent:
                sbu_append = [s for s in self.sbu_pool if s.name == sbu.parent][0]
                sbu_append.children.append(sbu)
                rem.append(id)
        for x in reversed(sorted(rem)):
            del self.sbu_pool[x]
    
    def _pop_unwanted_sbus(self):
        """Removes sbu indices not listed in the options."""
        remove = []
        # remove undesired organic SBUs
        [remove.append(x) for x, sbu in enumerate(self.sbu_pool) if
         sbu.identifier not in self.options.organic_sbus and
         not sbu.is_metal]
        # remove undesired metal SBUs
        [remove.append(x) for x, sbu in enumerate(self.sbu_pool) if
         sbu.identifier not in self.options.metal_sbus and
         sbu.is_metal]
        remove.sort()
        for p in reversed(remove):
            del self.sbu_pool[p]
                
        # issue warning if some of the SBUs requested in the ini file are not in the
        # database
        for sbu_request in self.options.organic_sbus:
            if sbu_request not in [i.identifier for i in self.sbu_pool if not i.is_metal]:
                warning("SBU id %i is not in the organic SBU database"%(int(sbu_request)))
        for sbu_request in self.options.metal_sbus:
            if sbu_request not in [i.identifier for i in self.sbu_pool if i.is_metal]:
                warning("SBU id %i is not in the metal SBU database"%(int(sbu_request)))
    def _pop_unwanted_topologies(self):
        [self._topologies.pop(k, None) for k in self._topologies.keys()
            if k not in self.options.topologies or k in 
            self.options.ignore_topologies]
        for k in self.options.topologies:
            if k not in self._topologies.keys():
                warning("Could not find the topology %s in the current "%(k) +
                       "database of topology files. Try including a file "+
                       "containing this topology to the input file.")

def main():
    options = config.Options()
    options.version = __version__
    log = glog.Log(options)
    jerb = JobHandler(options)
    jerb.direct_job()
    
if __name__ == '__main__':
    main()
