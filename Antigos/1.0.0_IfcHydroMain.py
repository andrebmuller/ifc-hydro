'''
Main in IfcHydro implementation.
'''

from datetime import datetime as time
from collections import defaultdict
import ifcopenshell as ifc

class Base:
    """ Base class. """
    
    _log     = "IfcHydro.log"        # name of log file
    _counter = 0                     # instance counter

    def __init__(self,log = ""):

        if log != "": 
            Base._log = log

        Base._counter += 1

    def append_log(self,text):
        """ Logging function. """

        t      = time.now()
        tstamp = "%2.2d.%2.2d.%2.2d " % (t.hour,t.minute,t.second)

        otext  = tstamp + text
        f = open(Base._log,"a")
        f.write(otext + "\n")
        f.close()
        print(otext)
    
    def resource_path(self, relative_path):
        """ Retrieves the relative path of resources used in the application. """
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

class Graph(object):
    """ Graph data structure, undirected by default. """

    def __init__(self, connections, directed=False):
        self._graph = defaultdict(set)
        self._directed = directed
        self.add_connections(connections)

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph. """

        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2):
        """ Add connection between node1 and node2. """

        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)

    def remove(self, node):
        """ Remove all references to node. """

        for n, cxns in self._graph.items():  # python3: items(); python2: iteritems()
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2. """

        return node1 in self._graph and node2 in self._graph[node1]

    def find_path(self, node1, node2, path=[]):
        """ Find any path between node1 and node2 (may not be shortest). """

        path = path + [node1]
        if node1 == node2:
            return path
        if node1 not in self._graph:
            return None
        for node in self._graph[node1]:
            if node not in path:
                new_path = self.find_path(node, node2, path)
                if new_path:
                    return new_path
        return None

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))

class TopologyCreator():
    """ Creates hydraulic system topology. """
    
    def __init__(self):
        """ Constructor. """
        self.model = ifc.open('ProjetoRamalBanheiro_Revit.ifc')

    def graph_creator(self):
        """ Creates graph with hydraulic system topology."""
        
        model = self.model

        Base.append_log(self,"> Model imported...")
        Base.append_log(self,"> Creating topology...")

        connections = []

        # Nest and connection lists
        nest_list = model.by_type("IfcRelNests")
        conn_list = model.by_type("IfcRelConnectsPorts")

        # Creates undirected graph with all nodes
        for conn in conn_list:
                for nest in nest_list:
                    for other_nest in nest_list:
                        if conn[4] in nest[5] or conn[5] in nest[5]:
                            if conn[5] in other_nest[5] or conn[4] in other_nest[5]:
                                nest1 = nest
                                nest2 = other_nest
                    
                    if nest1 != nest2:
                        connections.append((nest1[4], nest2[4]))
        
        graph = Graph(connections)

        return graph
    
    def path_finder(self, term_guid, tank_guid):
        """ Finds path between a sanitary terminal and a tank. """

        model = self.model
        graph = self.graph_creator()

        Base.append_log(self,"> Finding path...")
        path = []

        # Finds single path by two GUIDs
        term = model.by_guid(term_guid)
        tank = model.by_guid(tank_guid)

        path.append(graph.find_path(term, tank))
                
        return path

    def all_paths_finder(self):
        """ Finds path between all sanitary terminals and tanks. """

        model = self.model
        graph = self.graph_creator()

        Base.append_log(self,"> Finding path...")
        all_paths = []

        # Component lists
        term_list = model.by_type("IfcSanitaryTerminal")
        tank_list = model.by_type("IfcTank")

        # Appends all paths into all_paths list
        for term in term_list:
            for tank in tank_list:
                all_paths.append(graph.find_path(term, tank))
                
        return all_paths
    
class PropCalculator():
    """ Gets properties of the system components. """
    
    def __init__(self) -> None:
        """ Constructor. """
        pass

    def pipe_properties(self, pipe):
        """ Gets pipe properties from IFC file. """
        Base.append_log(self, f"> Getting pipe properties for pipe with ID {pipe.id()}...")
        pipe_prop = {}

        # Gets pipe length
        pipe_len = pipe[6][2][0][3][0][3]
        pipe_prop['len'] = round(pipe_len,3)

        # Gets pipe diameter
        pipe_dim = pipe[6][2][0][3][0][0][2][0][0][0][0]*2
        pipe_prop['dim'] = round(pipe_dim,3)

        Base.append_log(self, f"> Pipe properties:")
        Base.append_log(self, f"> {pipe_prop}")
        return(pipe_prop)
    
    def fitt_properties(self, fitt):
        """ Gets fittings properties from IFC file. """
        Base.append_log(self, f"> Getting fitting properties for fitting with ID {fitt.id()}...")
        fitt_prop = {}        

        # Reads hydraulic system topology
        topo = TopologyCreator()
        all_paths = topo.all_paths_finder()

        # Create path list with IDs
        all_paths_id = []
        n = 0
        for path in all_paths:
            all_paths_id.append([])
            for item in path:
                all_paths_id[n].append(all_paths[n][all_paths[n].index(item)].id())
            n += 1

        fitt_index = []
        for path in all_paths_id:
            for item in path:
                if item == fitt.id():
                    fitt_index.append(path.index(item))
            if fitt.id() not in path:
                fitt_index.append(None)

        pipe_list = []
        start_index = 0
        for item in fitt_index:
            if item is not None:
                pipe_list.append((all_paths[fitt_index.index(item, start_index)][item-1], all_paths[fitt_index.index(item, start_index)][item], all_paths[fitt_index.index(item)][item+1]))
                
                start_index = fitt_index.index(item)+1
            else:
                pipe_list.append(None)

        # Gets fitting diameters (entry and exits) by nearby pipes
        pipe_dim_list = []
        for item in pipe_list:
            if item is not None:
                pipe_dim_1 = pipe_list[pipe_list.index(item)][0][6][2][0][3][0][0][2][0][0][0][0]*2
                pipe_dim_2 = pipe_list[pipe_list.index(item)][2][6][2][0][3][0][0][2][0][0][0][0]*2
                pipe_dim_list.append((round(pipe_dim_1,3),round(pipe_dim_2,3)))
            else:
                pipe_dim_list.append(None)
                pass
        
        fitt_prop['dim'] = pipe_dim_list

        # Gets exit pipes vectors
        exit_vec_list = []
        for item in pipe_list:
            if item is not None:
                exit_pipe  = pipe_list[pipe_list.index(item)][0][6][2][0][3][0][1][0][0]
                fitt_vec   = pipe_list[pipe_list.index(item)][1][5][1][0][0]
                exit_vec_list.append((exit_pipe,fitt_vec))
            else:
                exit_vec_list.append(None)

        # Gets distance between outgoing pipe and fitting
        exit_res_vec_list = []
        for item in exit_vec_list:
            if item is not None:
                exit_res_vec_list.append(tuple((x-y) for x, y in zip(exit_vec_list[exit_vec_list.index(item)][0], exit_vec_list[exit_vec_list.index(item)][1])))
            else:
                exit_res_vec_list.append(None)

        # Gets direction of the vector between outgoing pipe and fitting
        exit_dir_list = []
        for item in exit_res_vec_list:
            exit_dir_tuple = ()
            if item is not None:
                for coordinate in item:
                    if round(coordinate,2) != 0:
                        exit_dir_tuple += (1, )
                    else:
                        exit_dir_tuple += (0, )
                exit_dir_list.append(exit_dir_tuple)
                del(exit_dir_tuple)
            else:
                exit_dir_list.append(None)


        fitt_prop['dir'] = exit_dir_list

        # Gets fitting type
        fitt_type = fitt[8]
        fitt_prop['type'] = fitt_type

        Base.append_log(self, f"> Fitting properties:")
        Base.append_log(self, f"> {fitt_prop}")
        return fitt_prop

    def valv_properties(self, valv):
        """ Gets valve properties from IFC file. """
        Base.append_log(self, f"> Getting valve properties for valve with ID {valv.id()}...")
        valv_prop = {}

        # Gets valve diameters (entry and exit) by nearby pipes
        topo = TopologyCreator()
        all_paths = topo.all_paths_finder()
        all_paths_id = []

        n = 0
        for path in all_paths:
            all_paths_id.append([])
            for item in path:
                all_paths_id[n].append(all_paths[n][all_paths[n].index(item)].id())
            n += 1

        valv_index = []
        for path in all_paths_id:
            for item in path:
                if item == valv.id():
                    valv_index.append(path.index(item))
            if valv.id() not in path:
                valv_index.append(None)

        pipe_list = []
        start_index = 0
        for item in valv_index:
            if item is not None:
                pipe_list.append((all_paths[valv_index.index(item, start_index)][item-1], all_paths[valv_index.index(item, start_index)][item], all_paths[valv_index.index(item)][item+1]))
                
                start_index = valv_index.index(item)+1
            else:
                pipe_list.append(None)

        pipe_dim_list = []
        for item in pipe_list:
            if item is not None:
                pipe_dim_1 = pipe_list[pipe_list.index(item)][0][6][2][0][3][0][0][2][0][0][0][0]*2
                pipe_dim_2 = pipe_list[pipe_list.index(item)][2][6][2][0][3][0][0][2][0][0][0][0]*2
                pipe_dim_list.append((round(pipe_dim_1,3),round(pipe_dim_2,3)))
            else:
                pipe_dim_list.append(None)
        
        valv_prop['dim'] = pipe_dim_list

        # Gets valve type
        valv_type = valv[8]
        valv_prop['type'] = valv_type

        Base.append_log(self, f"> Valve properties:")
        Base.append_log(self, f"> {valv_prop}")
        return valv_prop

class HydroCalculator():
    """ Makes hydraulical calculations of a system."""

    def __init__(self) -> None:
        pass

    def flow(self, all_paths):
        """ Calculates flow for every component in the system. """

        # Table with standarized design flow by sanitary terminal type
        design_flow_table = {'SHOWER': 0.2, 'WASHHANDBASIN': 0.15, 'WCSEAT': 0.15}

        # Creates a list with flow necessary for sanitary terminal on branch
        flow_list = []
        n = 0
        for path in all_paths:
            flow_list.append([])  
            i = -1          
            for component in path:
                if component.is_a() == "IfcSanitaryTerminal":
                    flow_list[n].append((component[0], design_flow_table[component[8]]))
                else:
                    flow_list[n].append((component[0], flow_list[n][i][1]))
                i += 1
            n += 1
        
        return flow_list                   

    def linear_pressure_drop(self, pipe, all_paths):
        """ Calculates the linear pressure drop in a certain pipe using the Hazen-Williams equation. """
        
        # Initializes variables
        prop_calc = PropCalculator()
        pipe_prop = prop_calc.pipe_properties(pipe)
        flow = self.flow(all_paths)
        design_flow = 0

        Base.append_log(self, f"> Getting linear pressure drop for pipe with ID {pipe.id()}...")

        # Calculates design flow for specified pipe
        for path in flow:
            for component in path:
                if component[0] == pipe[0]:
                    design_flow += component[1]
        
        # Calculates pressure drop using the Hazen-Williams equation
        # For PVC pipes only, C = 140
        pressure_drop = (10.67 * pipe_prop.get('len') * (design_flow * 0.001) ** 1.852) / ((140 ** 1.852) * (pipe_prop.get('dim') ** 4.87))

        Base.append_log(self, f"> Linear pressure drop:")
        Base.append_log(self, f"> {round(pressure_drop, 5)} m")
        return pressure_drop

    def local_pressure_drop(self, conn, all_paths):
        """ Gets the local pressure drop in a certain connection using table values. """

        local_pressure_drop_table = {'JUNCTION': 2.4, 'BEND': 1.2, 'EXIT': 1.2, 'ISOLATING': 0.2, 'REGULATING': 11.4}

        # Initializes variables
        prop_calc = PropCalculator()
        flow = self.flow(all_paths)
        design_flow = 0

        Base.append_log(self, f"> Getting local pressure drop for connection with ID {conn.id()}...")

        # Calculates design flow for specified connection
        for path in flow:
            for component in path:
                if component[0] == conn[0]:
                    design_flow += component[1]

        if conn.is_a() == 'IfcValve':
            conn_prop = prop_calc.valv_properties(conn)
        elif conn.is_a() == 'IfcPipeFitting':
            conn_prop = prop_calc.fitt_properties(conn)
        else:
            exit

        # Calculates pressure drop using the Hazen-Williams equation
        # For PVC connections only, C = 140
        pressure_drop = (10.67 * local_pressure_drop_table.get(conn_prop.get('type')) * (design_flow * 0.001) ** 1.852) / ((140 ** 1.852) * (0.025 ** 4.87))

        Base.append_log(self, f"> Local pressure drop:")
        Base.append_log(self, f"> {round(pressure_drop, 5)} m")
        return pressure_drop

    def available_pressure(self, term, all_paths):
        """ Calculates available pressure for a given sanitary terminal. """


        # Gets initial elevation difference (gravity potential)
        for path in all_paths:
            if term.id() == path[0].id():
                selected_path = path
                terminal_pipe_location  = path[0][6][2][0][3][0][1][0][0]
                tank_pipe_location = path[len(path)-2][6][2][0][3][0][1][0][0]

        pressure = (tank_pipe_location[2] + selected_path[len(selected_path)-2][5][0][1][0][0][2]) - terminal_pipe_location[2]

        Base.append_log(self, f"> Getting available pressure at sanitary terminal with ID {selected_path[0].id()}...")

        # Calculates pressure drop due to local and linear factors
        for component in selected_path:
            if component.is_a() == "IfcPipeSegment":
                pressure_drop = self.linear_pressure_drop(component, all_paths)
                pressure -= pressure_drop
            elif component.is_a() == "IfcPipeFitting":
                pressure_drop = self.local_pressure_drop(component, all_paths)
                pressure -= pressure_drop
            elif component.is_a() == "IfcValve":
                pressure_drop = self.local_pressure_drop(component, all_paths)
                pressure -= pressure_drop            
            else:
                pass

        Base.append_log(self, f"> Available pressure at the sanitary terminal:")
        Base.append_log(self, f"> {round(pressure,2)} m")
        return pressure

# Test environment
if __name__ == '__main__':

    topology = TopologyCreator()
    test_path = topology.all_paths_finder()

    model = ifc.open('ProjetoRamalBanheiro_Revit.ifc')
    prop_calc = PropCalculator()
    hydro_calc = HydroCalculator()

    """
    flow_calc_test = hydro_calc.flow(test_path)

    pipe_test = model.by_id(5399)
    pipe_prop_test = prop_calc.pipe_properties(pipe_test)
    pipe_calc_test = hydro_calc.linear_pressure_drop(pipe_test, test_path)

    fitt_test = model.by_id(7020)
    fitt_prop_test = prop_calc.fitt_properties(fitt_test)
    fitt_calc_test = hydro_calc.local_pressure_drop(fitt_test, test_path)

    valv_test = model.by_id(8087)
    valv_prop_test = prop_calc.valv_properties(valv_test)
    valv_calc_test = hydro_calc.local_pressure_drop(valv_test, test_path)    
    """
    
    term_test = model.by_id(5423)
    press_test = hydro_calc.available_pressure(term_test, test_path)
