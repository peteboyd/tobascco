#!/usr/bin/env sage-python 
import logging
import sys
from logging import info, debug, warning, error, critical
import config
from config import Terminate
import glog
import ConfigParser
from CSV import CSV
from Net import SystreDB, Net
from SecondaryBuildingUnit import SBU
#from CreateInput import SBUFileRead
from random import randint
import os

class JobHandler(object):
    """determines what job(s) to run based on arguments from the
    options class.
    
    """
    def __init__(self, options):
        self.options = options
        self._topologies = SystreDB()
        self.sbu_pool = []

    def direct_job(self):
        """Reads the options and decides what to do next."""
        
        # TODO(pboyd): problem reading in openbabel libraries for the inputfile
        #  creation due to the use of a custom python implemented for sage.


        #if self.options.create_sbu_input_files:
        #    info("Creating input files")
        #    job = SBUFileRead(self.options)
        #    job.read_sbu_files()
        #    job.sort_sbus()
        #    job.write_file()
        #    Terminate()

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
       
        print self.sbu_pool
        print self._topologies
        Terminate()
        for top in self.options.topologies:
            info("Starting with the topology: %s"%top)
            # delete all SBUs that are not listed in the ini file if
            # no specific combinations are requested.
            if not self.options.sbu_combinations:
                self._pop_unwanted_sbus()
            sbu_list = self._topologies[top]
            self._build_structures(sbu_list)

    def _build_structures(self, sbu_list):
        """Pass the sbu combinations to a MOF building algorithm."""
        if not sbu_list:
            info("No SBUs found!")
            return
        run = Generate(self.options, sbu_list)
        # generate the combinations of SBUs to build
        if self.options.sbu_combinations:
            combinations = run.combinations_from_options()
        else:
            # remove SBUs if not listed in options.organic_sbus or options.metal_sbus
            combinations = run.generate_sbu_combinations()

        # generate the MOFs.
        for combo in combinations:
            gen_counter = 0
            build = Build(self.options)
            extra = [j for i in combo for j in i.children]
            combo = tuple(list(combo) + extra)
            info("Trying %s"%(', '.join([i.name for i in combo])))
            if self.options.exhaustive:
                directives = run.generate_build_directives(None, combo)
            elif self.options.build_directives:
                directives = run.build_directives_from_options(build)

            for iter in range(self.options.max_trials):
                try:
                    d = directives.next()
                except StopIteration:
                    break
                # pass the directive to a MOF building algorithm
                gen = build.build_from_directives(d, combo)
                gen_counter = gen_counter + 1 if gen else gen_counter
                if gen_counter >= self.options.max_structures:
                    break
                # random increment if many trials have passed

                if iter >= (self.options.max_trials/2):
                    [directives.next() for i in range(randint(0,
                                    self.options.max_trials/3))]

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
            report.add_data(sbu_id = sbu.identifier)
            if self.options.calc_sbu_surface_area:
                report.add_data(surface_area = sbu.surface_area)
            if self.options.calc_max_sbu_span:
                report.add_data(sbu_span = sbu.max_span)
        
        # list organic SBUs second.
        for name, sbu in org_sbus.items():
            info("Computing data for %s"%name)
            report.add_data(sbu_id = sbu.identifier)
            if self.options.calc_sbu_surface_area:
                report.add_data(surface_area = sbu.surface_area)
            if self.options.calc_max_sbu_span:
                report.add_data(sbu_span = sbu.max_span)
        report.write()
         
    def _read_topology_database_files(self):
        for file in self.options.topology_files:
            db = SystreDB(filename=file)
            for top in db.keys():
                if top in self._topolgies.keys():
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
            self.sbu_pool.pop(x)
    
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
            self.sbu_pool.pop(p)
                
        # issue warning if some of the SBUs requested in the ini file are not in the
        # database
        for sbu_request in self.options.organic_sbus:
            if sbu_request not in [i.identifier for i in self.sbu_pool if not i.is_metal]:
                warning("SBU id %i is not in the organic SBU database"%(int(sbu_request)))
        for sbu_request in self.options.metal_sbus:
            if sbu_request not in [i.identifier for i in self.sbu_pool if i.is_metal]:
                warning("SBU id %i is not in the metal SBU database"%(int(sbu_request)))
    
def main():
    options = config.Options()
    log = glog.Log(options)
    jerb = JobHandler(options)
    jerb.direct_job()
    
if __name__ == '__main__':
    main()

