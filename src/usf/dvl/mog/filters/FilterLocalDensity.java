/*    Mapper On Graphs: A Mapper-based Topological Data Analysis tool for graphs 
 *    Copyright (C) Paul Rosen 2018-2019
 *    Additional Authors: Mustafa Hajij
 *    
 *    This program is free software: you can redistribute it and/or modify     
 *    it under the terms of the GNU General Public License as published by 
 *    the Free Software Foundation, either version 3 of the License, or 
 *    (at your option) any later version. 
 *     
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 *    GNU General Public License for more details.
 *    
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <https://www.gnu.org/licenses/>. 
*/
package usf.dvl.mog.filters;

import usf.dvl.graph.Graph;
import usf.dvl.tda.mapper.FilterFunction;

public class FilterLocalDensity extends FilterFunction {

	public FilterLocalDensity(Graph graph, float eps, int Type) {

		for (int i = 0; i < graph.getNodeCount(); i++) {
			Graph.GraphVertex v1 = graph.nodes.get(i);

			if (Type == 2) {
				double sum = 0;
				for (Graph.GraphVertex v : v1.getAdjacentVertices()) {
					for (Graph.GraphVertex v2 : v.getAdjacentVertices()) {
						if (!v1.isAdjacent(v2)) {
							sum += (double) Math.exp(-1 / eps); // assuming the distance between v and v1 is 1
						}
					}

				}
				put(v1, sum);
			}

			else {
				put(v1, (double) v1.getAdjacent().size() * (double) Math.exp(-1 / eps));
			}
		}

		finalize_init();
	}

	public String getName() { return "Local Density"; }
	public String getShortName() { return "Local Density"; }

}
