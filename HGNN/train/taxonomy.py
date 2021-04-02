import os
import pandas as pd

import pprint
pp = pprint.PrettyPrinter(indent=4)

# For phylogeny parsing
from opentree import OT
from ete3 import Tree

Fix_Tree = True


class Taxonomy:
    def __init__(self, node_ids, fileNameAndPath, verbose=False):
        self.ott_ids = []
        self.ott_id_dict = {}
        self.node_ids = node_ids
        self.fileNameAndPath = fileNameAndPath

        self.get_ott_ids(node_ids, verbose=verbose)
        self.get_tree(self.fileNameAndPath, self.ott_ids)


    def get_distance(self, species1, species2):
        # print(species1, species2)
        if species1 == species2:
            return 0

        ott_id1 = 'ott' + str(self.ott_id_dict[species1])
        ott_id2 = 'ott' + str(self.ott_id_dict[species2])
        # ott_id1 = self.ott_id_dict[species1]
        # ott_id2 = self.ott_id_dict[species2]
        # print(ott_id1, ott_id2)
        return self.tree.get_distance(ott_id1, ott_id2)

    # ------- privete functions

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
            df2['ott_id'] = df2.apply(lambda row: ott_id_dict[row['scientificName']], axis=1)
            # print(df)

        self.ott_ids = ott_ids
        self.ott_id_dict = ott_id_dict
        print(self.ott_id_dict)

    def fix_tree(self, fileNameAndPath):
        tree = Tree(fileNameAndPath, format=8)

        # Fix Esox Americanus
        D = tree.search_nodes(name="mrcaott47023ott496121")[0]
        D.name = "ott496115"
        tree.write(format=8, outfile=fileNameAndPath)
    
    def get_tree(self, fileNameAndPath, ott_ids):
        if not os.path.exists(fileNameAndPath):
            output = OT.synth_induced_tree(ott_ids=ott_ids, ignore_unknown_ids=False, label_format='id') # name_and_id ott_ids=list(ott_ids),

            output.tree.write(path = fileNameAndPath, schema = "newick")

            #TODO: fixing a species because it is actually a mid node.
            #TODO: Something to work with Yasin about
            if Fix_Tree:
                self.fix_tree(fileNameAndPath)

        

        self.tree = Tree(fileNameAndPath, format=8)
