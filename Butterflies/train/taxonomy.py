import os
import pandas as pd
import math

import pprint
pp = pprint.PrettyPrinter(indent=4)

# For phylogeny parsing
from opentree import OT
from ete3 import Tree, PhyloTree

Fix_Tree = True
format_ = 1 #8


class Taxonomy:
    def __init__(self, node_ids, fileNameAndPath, verbose=False):
        self.ott_ids = []
        self.ott_id_dict = {}
        self.node_ids = node_ids
        self.fileNameAndPath = fileNameAndPath
        self.total_distance = -1 # -1 means we never calculated it before.

        self.get_ott_ids(node_ids, verbose=verbose)
        self.get_tree(self.fileNameAndPath, self.ott_ids)
        self.get_total_distance()


    def get_distance(self, species1, species2):
        # print(species1, species2)
        if species1 == species2:
            return 0

        ott_id1 = 'ott' + str(self.ott_id_dict[species1])
        ott_id2 = 'ott' + str(self.ott_id_dict[species2])
        d = self.tree.get_distance(ott_id1, ott_id2)
        # ott_id1 = self.ott_id_dict[species1]
        # ott_id2 = self.ott_id_dict[species2]
        # print(ott_id1, ott_id2, d)
        return d
    
    def get_total_distance(self):
        # For one time, measure distance from all leaves down to root. They all should be equal.
        # Save the value and reuse it.
        
        if self.total_distance==-1:
            for leaf in self.tree.iter_leaves():
                total_distance = self.tree.get_distance(leaf) # gets distance to rootprint
                # print(total_distance)
                assert math.isclose(self.total_distance, total_distance) or self.total_distance==-1
                self.total_distance = total_distance

        return self.total_distance

    
    def get_siblings_by_name(self, species, relative_distance, get_ottids = False, verbose=False):
        ott_id = 'ott' + str(self.ott_id_dict[species])
        return self.get_siblings_by_ottid(ott_id, relative_distance, get_ottids, verbose)

    # ------- privete functions

    # relative_distance = 0 => species node itself
    # relative_distance = 1 => all species
    def get_siblings_by_ottid(self, ott_id, relative_distance, get_ottids = False, verbose=False):
        abs_distance = relative_distance*self.total_distance
        # for leaf in self.tree.iter_leaves():
        #     print(leaf.name)
        species_node = self.tree.search_nodes(name=ott_id)[0]
        if verbose:
            print('distance to common ancestor: ', abs_distance)
        # print('species', ott_id, species_node)

        # keep going up till distance exceeds abs_distance
        distance = 0
        parent = species_node
        while distance < abs_distance:
            parent = parent.up
            distance = self.tree.get_distance(parent, species_node)
            # print('distance', distance)
        
        # get all leaves under paernt
        node_list = parent.get_leaves()
        ott_id_list = list(map(lambda x: x.name, node_list))
        if verbose:
            print('species_list', len(ott_id_list),  ott_id_list)

        if not get_ottids:
            name_list = list(map(lambda x: next(key for key, value in self.ott_id_dict.items() if 'ott'+str(value) == x), ott_id_list)) # reverse lookup ott_id -> name
            return name_list
        

        return ott_id_list


    # node_ids: list of taxa
    # returns: corresponding list of ott_ids
    def get_ott_ids(self, node_ids, verbose=False):
        if verbose:
            print('Included taxonomy: ', node_ids, len(node_ids))
            df2 = pd.DataFrame(columns=['in csv', 'in response', 'Same?'])

        # Get the matches
        resp = OT.tnrs_match(node_ids, do_approximate_matching=True)
        matches = resp.response_dict['results']
        unmatched_names = resp.response_dict['unmatched_names']

        # Get the corresponding ott_ids
        ott_ids = set()
        ott_id_dict={}
        assert len(unmatched_names)==0 # everything is matched!
        for match_array in matches:
            match_array_matches = match_array['matches']
            assert len(match_array_matches)==1, match_array['name'] + " has too many matches" + str(list(map(lambda x: x['matched_name'], match_array_matches)))  # we have a single unambiguous match!
            first_match = match_array_matches[0]
            ott_id = first_match['taxon']['ott_id']
            ott_ids.add(ott_id)
            if verbose:
                #some original and matched names are not exactly the same. Not a bug
                df2 = df2.append({'in csv':match_array['name'], 'in response': first_match['matched_name'], 'Same?': match_array['name'] == first_match['matched_name']}, ignore_index=True)
                #print(match_array['name'], '\t -> \t', first_match['matched_name']) 
            ott_id_dict[match_array['name']] = ott_id
        ott_ids = list(ott_ids)

        if verbose:
            print(df2[df2['Same?']== False])
            # print(ott_ids, len(ott_ids))
            pp.pprint(ott_id_dict)

            # add ott_id to metadata csv file
            # df2['ott_id'] = df2.apply(lambda row: ott_id_dict[row['scientificName']], axis=1)
            # print(df)

        self.ott_ids = ott_ids
        self.ott_id_dict = ott_id_dict
        print(self.ott_id_dict)

    def fix_tree(self, fileNameAndPath):
        tree = PhyloTree(fileNameAndPath, format=format_)

        # Fix Esox Americanus
        D = tree.search_nodes(name="mrcaott47023ott496121")[0]
        D.name = "ott496115"
        tree.write(format=format_, outfile=fileNameAndPath)
    
    def get_tree(self, fileNameAndPath, ott_ids):
        if not os.path.exists(fileNameAndPath):
            output = OT.synth_induced_tree(ott_ids=ott_ids, ignore_unknown_ids=False, label_format='id') # name_and_id ott_ids=list(ott_ids),

            output.tree.write(path = fileNameAndPath, schema = "newick")

            #TODO: fixing a species because it is actually a mid node.
            #TODO: Something to work with Yasin about
            if Fix_Tree:
                self.fix_tree(fileNameAndPath)

        

        self.tree = PhyloTree(fileNameAndPath, format=format_)
