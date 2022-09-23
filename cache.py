import json
import os
import sys
import time

import mog.mapper as mapper
import mog.graph_io as GraphIO


def get_graph_path( params ):
    return 'docs/data/' + params['datafile'] + ".json"


def save_graph_layout(params, data):
    filename = get_graph_path(params)
    print(filename)
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def get_filter_function(params):
    rank_filter = False if 'rank_filter' not in params else params['rank_filter'].lower() == 'true'
    filename = os.path.splitext(params['datafile'])[0] + "/" + params['filter_func'] + ".json"
    print(filename)
    return GraphIO.read_filter_function('docs/data/' + filename, rank_filter)


def get_mog_path(datafile, ff, params=None):
    path = 'docs/cache/' + os.path.splitext(datafile)[0] + '/' + ff
    if params is not None:
        keys = list(params.keys())
        keys.sort()
        for k in keys:
            path += "_" + str(params[k])
    return path + ".json"


def generate_mog(datafile, filter_func, cover_elem_count, cover_overlap, comp_method, link_method, rank_filter):
    mog = mapper.MapperOnGraphs()

    mog_cf = get_mog_path(datafile, filter_func, {
                                                        'coverN': cover_elem_count,
                                                        'coverOverlap': cover_overlap,
                                                        'component_method': comp_method,
                                                        'link_method': link_method,
                                                        'rank_filter': 'false' if rank_filter is None else rank_filter
                                                    })

    if os.path.exists(mog_cf):
        print("  >> found " + mog_cf + " in cache")
        try:
            mog.load_mog(mog_cf)
            return mog, mog_cf
        except:
            print("    failed to load -- " + str(sys.exc_info()[0]))

    print("  >> creating " + mog_cf)

    create_dir = ""
    for d in mog_cf.split('/')[:-1]:
        create_dir += d + '/'
        if not os.path.exists(create_dir): os.mkdir(create_dir)

    # Load the graph and filter function
    start_time = time.time()
    graph_data, graph = GraphIO.read_json_graph('docs/data/' + datafile + ".json")

    values = get_filter_function({'datafile': datafile, 'filter_func': filter_func,
                                  'rank_filter': rank_filter})
    end_time = time.time()

    print(" >> Input Load Time: " + str(end_time-start_time))
    print(" >> Input Node Count: " + str(graph.number_of_nodes()))
    print(" >> Input Edge Count: " + str(graph.number_of_nodes()))

    # Construct the cover
    intervals = int(cover_elem_count)
    overlap = float(cover_overlap)
    cover = mapper.Cover(values, intervals, overlap)

    # Construct & save MOG
    mog.build_mog(graph, values, cover, comp_method, link_method, verbose=graph.number_of_nodes() > 1000)
    mog.strip_components_from_nodes()
    mog.save_json(mog_cf)

    print(" >> MOG Node Count: " + str(mog.number_of_nodes()))
    print(" >> MOG Edge Count: " + str(mog.number_of_nodes()))
    print(" >> MOG Compute Time: " + str(mog.compute_time()) + " seconds")

    return mog, mog_cf


def get_mog(params):
    mog, mog_cf = generate_mog(params['datafile'],
                               params['filter_func'],
                               params['coverN'], params['coverOverlap'],
                               params['component_method'],
                               params['link_method'], params['rank_filter'])

    node_size_filter = int(params['mapper_node_size_filter'])
    if node_size_filter > 0:
        mog.filter_node_size(node_size_filter)

    if params['gcc_only'] == 'true':
        mog.extract_greatest_connect_component()

    return mog.to_json()
