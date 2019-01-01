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
package usf.dvl.graph.mapper.filter;

import java.util.List;

import org.ejml.simple.SimpleMatrix;

import usf.dvl.graph.Graph;
import usf.dvl.graph.LaplacianMatrix.Eigen;


public class FilterEigenFunctions extends Filter {

	int which;
	
	public FilterEigenFunctions( Graph graph, int whichone) {
		which = whichone;

		List<Eigen> alleig = graph.toGraphLaplacian().eigen();
		SimpleMatrix eig = alleig.get(whichone).getEigenVector();

		for (int i = 0; i < graph.getNodeCount(); i++ ) {
			Graph.GraphVertex v1 = graph.nodes.get(i);
			put(v1, eig.get(i));  
		}

		finalize();
	}

	public String getName() { return "Eigen Function " + which; }
	public String getShortName() { return "Eigen Func " + which; }
	
}
