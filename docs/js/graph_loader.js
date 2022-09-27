


function get_default_mog_options(ds,df,ff,coverN=3,rank=false,component='modularity'){
    return {
            'datafile': df,
            'filter_func': ff,
            'coverN': coverN,
            'rank_filter':  rank,
            'coverOverlap': 0,
            'component_method': component,
            'link_method': 'connectivity',
            'gcc_only':  false,
            'mapper_node_size_filter': 0
    }
}



let Load_MOG_Vis = function(svg_name, options, static_uri=true, end_cb = ()=>{} ){
    let param_uri = $.param(options)
    let mog_data = null
    let mog_vis = null
    let uri = 'mog?' + param_uri
    let ff = options.filter_func

    let svg = d3.select(svg_name)
    svg.style("cursor", "progress" )

    if( static_uri ){
        let df = options.datafile
        //df = df.substring(0,df.length-5)
        let covN = options.coverN
        let link_m = options.link_method
        let rank = ( options.rank_filter ) ? 'true' : 'false'
        let comp_m = options.component_method
        let overlap = options.coverOverlap

        uri = 'cache/' + df + '/' + ff + '_' + comp_m
                + '_' + covN + '_' + overlap + '_' + link_m + '_' + rank + '.json'
    }

    d3.json(uri, function (error, data) {
        if (error) {
            console.log(error);
        } else {
            mog_data = data
            mog_vis = GraphVisualization(svg_name, mog_data);
            mog_vis.set_node_radius(d => Math.min( Math.sqrt(d.comp_len)*2.5, 20));
            mog_vis.set_node_color(n=>colorSchemes[ff](n.avg_v));
            mog_vis.set_link_width(d => Math.min( Math.sqrt(d.value), 20)/2 );
            mog_vis.add_count_labels();
            mog_vis.set_tick_callback( (n) => {
                if(n===10) mog_vis.zoomFit();
                if(n===100) mog_vis.zoomFit();
            })
            mog_vis.set_end_callback( ()=>{
                mog_vis.zoomFit();
                mog_vis.send_to_url( "update_mog?"+param_uri, {'info':mog_data.info} )
                end_cb()
            });
            mog_vis.load()
        }
        svg.style("cursor", "auto" )
    });

    return {
        remove : function(){
            if( mog_vis ) mog_vis.remove();
        },
        get_graph : function(){
            return mog_vis
        }
    }

}


let Load_Graph_Vis = function(svg_name, options, static_uri=true, cb = null, end_cb = ()=>{} ){

    let uri = 'graph?' + $.param(options)
    if( static_uri ) uri = 'data/' + options.datafile + '.json'

    let svg = d3.select(svg_name)
    svg.style("cursor", "progress" )

    let graph_data = null
    let graph_vis = null

    d3.json(uri, function (error, data) {
        if (error) {
            console.log(error);
        } else {
            graph_data = data
            graph_vis = GraphVisualization(svg_name, data);
            graph_vis.set_tick_callback( (n) => {
                if(n===10) graph_vis.zoomFit();
                if(n===100) graph_vis.zoomFit();
            })
            graph_vis.set_end_callback( ()=>{
                graph_vis.zoomFit();
                graph_vis.send_to_url( "update_graph?" + $.param(options) )
                end_cb()
            } );
            graph_vis.load();
        }
        svg.style("cursor", "auto" )
        if( cb ) cb(graph_vis)
    });

    return {
        remove : function(){
            if( graph_vis ) graph_vis.remove();
        },
        set_node_color : function( func, data ){
            if(graph_vis) graph_vis.set_node_color(func, data);
        }
    }
}

let Load_Graph_Vis_W_FF = function(svg_name, options, static_uri = true, cb=null, end_cb = ()=>{} ){
    Load_Filter_Function(options, (ff_data)=>{
        Load_Graph_Vis( svg_name, options, true, (gv )=>{
            let cs = colorSchemes[options.filter_func].domain([0,1])
            gv.set_node_color( n => cs(n), ff_data.data )
            if(cb) cb(gv, ff_data)
        }, end_cb )
    })
}

let Load_Filter_Function = function( options, cb, static_uri=true ){

    let uri = "filter_function?" + $.param(options)

    if (static_uri) {
        let df = options.datafile //.substring(0,options.datafile.length-5)
        uri = 'data/' + df + '/' + options.filter_func + '.json'
        console.log(df)
    }

    d3.json(uri, function (error, _ff_data) {
        if (error) {
            console.log(error);
        } else {
            cb(_ff_data)
        }
    })
}


















function __send_graph_to_cache(url, graph_data){
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    tmp_data = {'nodes':graph_data.nodes,'links':[]};
    graph_data.links.forEach( function(L){
       tmp_data.links.push({'value':L.value,'source':L.source.id,'target':L.target.id});
    });
    xhr.send(JSON.stringify(tmp_data));
}

// let MOG_Vis = function( svg_name, options ){
//     let param_uri = $.param(options)
//     let mog_data = null
//     let mog_vis = null
//
//     d3.json("mog?" + param_uri, function (error, data) {
//         if (error) {
//             console.log(error);
//         } else {
//             mog_data = data
//             mog_vis = FDL_Graph_Vis(svg_name, mog_data);
//             mog_vis.set_node_radius(d => Math.min( Math.sqrt(d.component_count), 20));
//             mog_vis.set_node_color(n=>colorSchemes[options.filter_func](n.avg_value));
//             mog_vis.set_link_width(d => Math.min( Math.sqrt(d.value), 20)/2 );
//             //mog_vis.add_count_labels();
//             mog_vis.set_end_callback(()=>{
//                 mog_vis.zoomFit();
//                 __send_graph_to_cache("cache?type=mog_layout&"+param_uri, mog_data)
//             });
//             mog_vis.load()
//         }
//     });
//
// }

let Graph_VIS = function(svg_name, options, show_ff = false){
    let param_uri = $.param(options)
    let graph_vis = null
    let graph_data = null
    let ff_data = null


    d3.json("graph?" + param_uri, function (error, _g_data) {
        if (error) {
            console.log(error);
        } else {
            graph_data = _g_data
            if( show_ff ) {
                d3.json("filter_function?" + param_uri, function (error, _ff_data) {
                    ff_data = _ff_data
                    let cs = colorSchemes[options.filter_func].domain([0, 1])
                    graph_vis = GraphVisualization(svg_name, graph_data);
                    graph_vis.set_node_radius( ()=> 3)
                    graph_vis.set_end_callback(()=>{
                        graph_vis.zoomFit()
                        __send_graph_to_cache("cache?type=graph_layout&"+param_uri, graph_data)
                    });
                    graph_vis.set_node_color(n => cs(n), ff_data);
                    graph_vis.load();
                });
            }
            else{
                graph_vis = GraphVisualization(svg_name, data);
                graph_vis.set_node_radius( ()=> 3)
                graph_vis.set_end_callback(()=>{
                    graph_vis.zoomFit()
                    __send_graph_to_cache("cache?type=graph_layout&"+param_uri, graph_data)
                });
                graph_vis.load();
            }

        }
    });
}
