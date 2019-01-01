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

import processing.core.PApplet;
import usf.dvl.graph.Graph;

public class FilterFromFile extends Filter {

	public FilterFromFile( PApplet p, Graph graph, String filename) {

		String [] lines = p.loadStrings( filename );

		for( int i = 1; i < lines.length; i++ ){
			String [] parts = lines[i].split("\\s+");
			if( parts.length >= 2 ){
				int    vid = Integer.parseInt(parts[0]);
				double val = Double.parseDouble(parts[1]);
				put( graph.nodes.get(vid), val);
			}
		}

		finalize();
	}

	public String getName() { return "Loaded From File"; }
	public String getShortName() { return "From File"; }

}