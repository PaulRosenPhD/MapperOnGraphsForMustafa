import fnmatch
import json
import multiprocessing
import ntpath
import os
import sys
import time
import networkx as nx

import mog.graph_io as GraphIO
import mog.filter_functions as ff
import layout.initial_layout as layout

import cache

data_sets = {}

filter_function_names = {'agd': 'Average Geodesic Distance',
                         'ecc': 'Eccentricity',
                         'pr_0_85': 'PageRank (alpha=0.85)',
                         'fv': 'Fiedler Vector',
                         'fv_norm': 'Fiedler Vector Normalized',
                         'den_0_5': 'Density 0.5'}


def process_graph(in_filename):
    print("Found: " + in_filename)
    basename, ext = os.path.splitext(ntpath.basename(in_filename).lower())
    out_filename = 'docs/' + os.path.splitext(in_filename.lower())[0] + '.json'

    if os.path.exists(out_filename): return out_filename

    graph=None
    graphs=None
    if ext == ".json": data, graph = GraphIO.read_json_graph(in_filename)
    elif ext == ".graph": data, graph = GraphIO.read_graph_file(in_filename)
    elif ext == ".tsv": data, graph = GraphIO.read_tsv_graph_file(in_filename)
    elif ext == ".npz": data, graphs = GraphIO.read_numpy_graphs(in_filename)
    elif ext == '.obj': data, graph = GraphIO.read_obj_graph(in_filename)
    else: return None

    if graph is None and graphs is None: return None

    create_dir = ""
    for d in out_filename.split('/')[:-1]:
        create_dir += d + '/'
        if not os.path.exists(create_dir): os.mkdir(create_dir)

    print("   >> Converting " + in_filename + " to " + out_filename)

    if graph is not None:
        # Extract the largest connected component
        gcc = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(gcc)

        # Provide a good quality initial layout for small and medium sized graphs
        if graph.number_of_nodes() < 5000:
            layout.initialize_radial_layout(graph)

        # # Write the graph to file
        GraphIO.write_json_graph(out_filename, graph)

        return [out_filename]
    else:
        ofs = []
        for i,g in enumerate(graphs['train']):
            # print(i,g)
            fn = out_filename[:-5] + '_train_' + str(i) + '.json'
            # print(fn)
            ofs.append(fn)
            GraphIO.write_json_graph( fn, g )
        return ofs


# Generate AGD
def generate_agd(out_path, graph, weight):
    if not os.path.exists(out_path):
        print("   >> Generating AGD")
        data = ff.average_geodesic_distance(graph, _weight=weight, _out_path=out_path)


# Generate eccentricity
def generate_ecc(out_path, graph):
    if not os.path.exists(out_path):
        print("   >> Generating Eccentricity")
        data = ff.eccentricity(graph, _out_path=out_path)


# Generate pagerank
def generate_pr(out_path, graph, weight, alpha):
    if not os.path.exists(out_path):
        print("   >> Generating Pagerank")
        data = ff.pagerank(graph, weight, alpha, _out_path=out_path)


# Generate fiedler vector
def generate_fv(out_path, graph, weight, normalized):
    if not os.path.exists(out_path):
        print("   >> Generating Fiedler Vector")
        data = ff.fiedler_vector(graph, _weight=weight, _normalized=normalized, _out_path=out_path)


# Generate density
def generate_den(out_path, graph, weight, eps):
    if not os.path.exists(out_path):
        print("   >> Generating Density")
        data = ff.density(graph, weight, eps, _out_path=out_path)


# Function that controls the creating of filter functions
def process_filter_functions(in_filename, max_time_per_file=1, scalableOnly=False):
    print("Processing Graph: " + in_filename)

    out_dir = os.path.splitext(in_filename)[0]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    need_processing = False
    for f in filter_function_names.keys():
        need_processing = need_processing or not os.path.exists(out_dir + '/' + f + ".json")

    if not need_processing: return

    data, graph = GraphIO.read_json_graph(in_filename)

    mprocs = [multiprocessing.Process(target=generate_agd, args=(out_dir + "/agd.json", graph, 'value')),
                 multiprocessing.Process(target=generate_ecc, args=(out_dir + "/ecc.json", graph)),
                 multiprocessing.Process(target=generate_pr, args=(out_dir + "/pr_0_85.json", graph, 'value', 0.85)),
                 multiprocessing.Process(target=generate_fv, args=(out_dir + "/fv.json", graph, 'value', False)),
                 multiprocessing.Process(target=generate_fv, args=(out_dir + "/fv_norm.json", graph, 'value', True)),
                 multiprocessing.Process(target=generate_den, args=(out_dir + "/den_0_5.json", graph, 'value', 0.5))]

    # process the functions in parallel for max_time_per_file
    end_time = time.time() + max_time_per_file
    for p in mprocs: p.start()
    for p in mprocs:
        p.join(max(1, int(end_time - time.time())))
        if p.is_alive():
            p.terminate()
            p.join()


def generate_data(max_time_per_file=1):

    data_gen = []
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith(".npz") or file.endswith(".obj"):
                data_gen.extend( process_graph(os.path.join(root, file) ) )
    
    for file in data_gen:
        if file is None: continue
        try:
            process_filter_functions(file, max_time_per_file)
        except json.decoder.JSONDecodeError:
            print(">>> FAILED: json parse " + file)
        except TypeError:
            print(">>> FAILED: type error " + file)
        except nx.exception.NetworkXError:
            print(">>> FAILED: graph not connected error " + file)
        except:
            print(file + " failed with " + str(sys.exc_info()[0]))


def scan_datasets():
    # for d0 in ['very_small', 'small', 'medium', 'large']:
    for root, dirs, files in os.walk('docs/data'):
        troot = root[9:]
        for file in files:
            if file[:-5] in filter_function_names.keys():
                if troot not in data_sets: data_sets[troot] = {}
                data_sets[troot][file[:-5]] = filter_function_names[file[:-5]]
    GraphIO.write_json_data('docs/data/datasets.json',data_sets)


def __pre_generate_mog( params, opts, opts_keys ):
    if len(opts_keys) == 0:
        cache.generate_mog(params['datafile'],
                           params['filter_func'],
                           params['coverN'], params['coverOverlap'],
                           params['component_method'],
                           params['link_method'], params['rank_filter'])
    else:
        key = opts_keys[0]
        for o in opts[key]:
            params[key] = o
            __pre_generate_mog(params, opts, opts_keys[1:])


def pre_generate_mog(datafile,ff):
    opts = {
        'datafile': [datafile],
        'filter_func': ff,
        'coverN': [2,3,4,6,8,10,20],
        'coverOverlap': [0],
        'component_method': ['connected_components','modularity','async_label_prop'],
        'link_method': ['connectivity'],
        'mapper_node_size_filter': [0],
        'rank_filter': ['true','false'],
        'gcc_only': ['false']
    }
    __pre_generate_mog( {}, opts, list(opts.keys()) )


if __name__ == '__main__':

    timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    if timeout > 0:
        generate_data(timeout)
    
    scan_datasets()
    
    with multiprocessing.Pool(processes=6) as pool:
        procs = []
        for d0 in data_sets:
            procs.append( pool.apply_async(pre_generate_mog, (d0,data_sets[d0]) ) )
        [res.get(timeout=900) for res in procs]
