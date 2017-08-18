import os
import networkx as nx
import math
import numpy as np
import random
import scipy.stats as stats
import tqdm
import sys
import past.builtins
import future.utils

__author__ = 'Giulio Rossetti'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com"
__version__ = "0.2.0"


class RDyn(object):

    def __init__(self, size=1000, iterations=1000, avg_deg=15, sigma=.7,
                 lambdad=1, alpha=3, paction=1, prenewal=.8,
                 quality_threshold=.3, new_node=.0, del_node=.0, max_evts=1):

        # set the network generator parameters
        self.size = size
        self.iterations = iterations
        self.avg_deg = avg_deg
        self.sigma = sigma
        self.lambdad = lambdad
        self.exponent = alpha
        self.paction = paction
        self.renewal = prenewal
        self.new_node = new_node
        self.del_node = del_node
        self.max_evts = max_evts

        # event targets
        self.communities_involved = []

        # initialize communities data structures
        self.communities = {}
        self.node_to_com = []
        self.total_coms = 0
        self.performed_community_action = "START\n"
        self.quality_threshold = quality_threshold
        self.exp_node_degs = []

        # initialize the graph
        self.graph = nx.empty_graph(self.size)

        self.base = os.getcwd()

        # initialize output files
        self.output_dir = "%s%sresults%s%s_%s_%s_%s_%s_%s_%s" % \
            (self.base, os.sep, os.sep, self.size, self.iterations, self.avg_deg, self.sigma, self.renewal, self.quality_threshold, self.max_evts)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.out_interactions = open("%s%sinteractions.txt" % (self.output_dir, os.sep), "w")
        self.out_events = open("%s%sevents.txt" % (self.output_dir, os.sep), "w")
        self.stable = 0

        self.it = 0
        self.count = 0

    def __get_assignation(self, community_sizes):

        degs = [(i, self.exp_node_degs[i]) for i in past.builtins.xrange(0, len(self.exp_node_degs))]

        for c in past.builtins.xrange(0, len(community_sizes)):
            self.communities[c] = []

        unassigned = []

        for n in degs:
            nid,  nd = n
            assigned = False
            for c in past.builtins.xrange(0, len(community_sizes)):
                c_size = community_sizes[c]
                c_taken = len(self.communities[c])
                if c_size/float(nd) >= self.sigma and c_taken < c_size:
                    self.communities[c].append(nid)
                    assigned = True
                    break
            if not assigned:
                unassigned.append(n)

        slots_available = [(k, (community_sizes[k] - len(self.communities[k]))) for k in past.builtins.xrange(0, len(community_sizes))
                           if (community_sizes[k] - len(self.communities[k])) > 0]

        if len(unassigned) > 0:
            for i in unassigned:
                for c in past.builtins.xrange(0, len(slots_available)):
                    cid, av = slots_available[c]
                    if av > 0:
                        self.communities[cid].append(i[0])
                        self.exp_node_degs[i[0]] = community_sizes[cid] - 1
                        slots_available[c] = (cid, av-1)
                        break

        ntc = {}
        for cid, nodes in future.utils.iteritems(self.communities):
            for n in nodes:
                ntc[n] = cid

        nodes = ntc.keys()
        nodes.sort()

        for n in nodes:
            self.node_to_com.append(ntc[n])

    @staticmethod
    def __truncated_power_law(alpha, maxv, minv=1):
        """

        :param maxv:
        :param minv:
        :param alpha:
        :return:
        :rtype: object
        """
        x = np.arange(1, maxv + 1, dtype='float')
        pmf = 1 / x ** alpha
        pmf /= pmf.sum()
        ds = stats.rv_discrete(values=(range(minv, maxv + 1), pmf))

        return ds

    def __add_node(self):
        nid = self.size
        self.graph.add_node(nid)
        cid = random.sample(self.communities.keys(), 1)[0]
        self.communities[cid].append(nid)
        self.node_to_com.append(cid)
        deg = random.sample(range(2, int((len(self.communities[cid])-1) +
                                  (len(self.communities[cid])-1)*(1-self.sigma))), 1)[0]
        if deg == 0:
            deg = 1
        self.exp_node_degs.append(deg)
        self.size += 1

    def __remove_node(self):

        com_sel = [c for c, v in future.utils.iteritems(self.communities) if len(v) > 3]
        if len(com_sel) > 0:
            cid = random.sample(com_sel, 1)[0]
            s = self.graph.subgraph(self.communities[cid])
            min_value = min(s.degree().itervalues())
            candidates = [k for k in s.degree() if s.degree()[k] == min_value]
            nid = random.sample(candidates, 1)[0]
            for e in self.graph.edges([nid]):
                self.count += 1
                self.out_interactions.write("%s\t%s\t-\t%s\t%s\n" % (self.it, self.count, e[0], e[1]))
                self.graph.remove_edge(e[0], e[1])

            self.exp_node_degs[nid] = 0
            self.node_to_com[nid] = -1
            nodes = set(self.communities[cid])
            self.communities[cid] = list(nodes - {nid})
            self.graph.remove_node(nid)

    def __get_degree_sequence(self):
        """

        :return:

        :rtype: object
        """
        minx = float(self.avg_deg) / (2 ** (1 / (self.exponent - 1)))

        while True:
            exp_deg_dist = self.__truncated_power_law(self.exponent, self.size, int(math.ceil(minx)))
            degs = list(exp_deg_dist.rvs(size=self.size))

            if nx.is_valid_degree_sequence(degs):
                return degs, int(minx)

    def __test_communities(self):
        mcond = 0
        for k in self.communities_involved:
            c = self.communities[k]
            if len(c) == 0:
                return False

            s = self.graph.subgraph(c)
            comps = nx.number_connected_components(s)

            if comps > 1:
                cs = nx.connected_components(s)
                i = random.sample(cs.next(), 1)[0]
                j = random.sample(cs.next(), 1)[0]
                timeout = (self.it + 1) + int(random.expovariate(self.lambdad))
                self.graph.add_edge(i, j, {"d": timeout})
                self.count += 1
                self.out_interactions.write("%s\t%s\t+\t%s\t%s\n" % (self.it, self.count, i, j))
                return False

            score = self.__conductance_test(k, s)
            if score > mcond:
                mcond = score
        if mcond > self.quality_threshold:
            return False

        return True

    def __conductance_test(self, comid, community):
        s_degs = community.degree()
        g_degs = self.graph.degree(community.nodes())

        # Conductance
        edge_across = 2 * sum([g_degs[n] - s_degs[n] for n in community.nodes()])
        c_nodes_total_edges = community.number_of_edges() + (2 * edge_across)

        if edge_across > 0:
            ratio = float(edge_across) / float(c_nodes_total_edges)
            if ratio > self.quality_threshold:
                self.communities_involved.append(comid)
                self.communities_involved = list(set(self.communities_involved))

                for i in community.nodes():
                    nn = self.graph.neighbors(i)
                    for j in nn:
                        if j not in community.nodes():
                            self.count += 1
                            self.out_interactions.write("%s\t%s\t-\t%s\t%s\n" % (self.it, self.count, i, j))
                            self.graph.remove_edge(i, j)
                            continue
            return ratio
        return 0

    def __generate_event(self, simplified=True):
        communities_involved = []
        self.stable += 1

        options = ["M", "S"]

        evt_number = random.sample(range(1, self.max_evts+1), 1)[0]
        evs = np.random.choice(options, evt_number, p=[.5, .5], replace=True)
        chosen = []

        if len(self.communities) == 1:
            evs = "S"

        self.__output_communities()
        if "START" in self.performed_community_action:
            self.out_events.write("%s:\t%s" % (self.it, self.performed_community_action))
        else:
            self.out_events.write("%s:\n%s" % (self.it, self.performed_community_action))

        self.performed_community_action = ""

        for p in evs:

            if p == "M":
                # Generate a single merge
                if len(self.communities) == 1:
                    continue
                candidates = list(set(self.communities.keys()) - set(chosen))

                # promote merging of small communities
                cl = [len(v) for c, v in future.utils.iteritems(self.communities) if c in candidates]
                comd = 1-np.array(cl, dtype="float")/sum(cl)
                comd /= sum(comd)

                ids = []
                try:
                    ids = np.random.choice(candidates, 2, p=list(comd), replace=False)
                except:
                    continue

                # ids = random.sample(candidates, 2)
                chosen.extend(ids)
                communities_involved.extend([ids[0]])

                for node in self.communities[ids[1]]:
                    self.node_to_com[node] = ids[0]

                self.performed_community_action = "%s MERGE\t%s\n" % (self.performed_community_action, ids)

                self.communities[ids[0]].extend(self.communities[ids[1]])
                del self.communities[ids[1]]

            else:
                # Generate a single splits
                if len(self.communities) == 1:
                    continue

                candidates = list(set(self.communities.keys()) - set(chosen))

                cl = [len(v) for c, v in future.utils.iteritems(self.communities) if c in candidates]
                comd = np.array(cl, dtype="float")/sum(cl)

                try:
                    ids = np.random.choice(candidates, 1, p=list(comd), replace=False)
                except:
                    continue

                c_nodes = len(self.communities[ids[0]])
                if c_nodes > 6:
                    try:
                        size = random.sample(range(3, c_nodes-3), 1)[0]
                        first = random.sample(self.communities[ids[0]], size)
                    except:
                        continue
                    cid = max(self.communities.keys()) + 1
                    chosen.extend([ids[0], cid])
                    communities_involved.extend([ids[0], cid])

                    self.performed_community_action = "%s SPLIT\t%s\t%s\n" % \
                                                      (self.performed_community_action, ids[0], [ids[0], cid])
                    # adjusting max degree
                    for node in first:
                        self.node_to_com[node] = cid
                        if self.exp_node_degs[node] > (len(first)-1) * self.sigma:
                            self.exp_node_degs[node] = int((len(first)-1) + (len(first)-1) * (1-self.sigma))

                    self.communities[cid] = first
                    self.communities[ids[0]] = [ci for ci in self.communities[ids[0]] if ci not in first]

                    # adjusting max degree
                    for node in self.communities[ids[0]]:
                        if self.exp_node_degs[node] > (len(self.communities[ids[0]])-1) * self.sigma:
                            self.exp_node_degs[node] = int((len(self.communities[ids[0]])-1) +
                                                           (len(self.communities[ids[0]])-1) * (1-self.sigma))

        self.out_events.flush()
        if not simplified:
            communities_involved = self.communities.keys()

        return self.node_to_com, self.communities, communities_involved

    def __output_communities(self):

        self.total_coms = len(self.communities)
        out = open("%s%scommunities-%s.txt" % (self.output_dir, os.sep, self.it), "w")
        for c, v in future.utils.iteritems(self.communities):
            out.write("%s\t%s\n" % (c, v))
        out.flush()
        out.close()

        outg = open("%s%sgraph-%s.txt" % (self.output_dir, os.sep, self.it), "w")
        for e in self.graph.edges():
            outg.write("%s\t%s\n" % (e[0], e[1]))
        outg.flush()
        outg.close()

    def __get_community_size_distribution(self, mins=3):
        cv, nc = 0, 2
        cms = []

        nc += 2
        com_s = self.__truncated_power_law(2, self.size/self.avg_deg, mins)
        exp_com_s = com_s.rvs(size=self.size)

        # complete coverage
        while cv <= 1:

            cms = random.sample(exp_com_s, nc)
            cv = float(sum(cms)) / self.size
            nc += 1

        while True:
            if sum(cms) <= self.size:
                break

            for cm in past.builtins.xrange(-1, -len(cms), -1):
                if sum(cms) <= self.size:
                    break
                elif sum(cms) > self.size and cms[cm] > mins:
                    cms[cm] -= 1

        return sorted(cms, reverse=True)

    def execute(self, simplified=True):
        """

        :return:
        """
        if self.size < 1000:
            print("Minimum network size: 1000 nodes")
            exit(0)

        # generate pawerlaw degree sequence
        self.exp_node_degs, mind = self.__get_degree_sequence()

        # generate community size dist
        exp_com_s = self.__get_community_size_distribution(mins=mind+1)

        # assign node to community
        self.__get_assignation(exp_com_s)

        self.total_coms = len(self.communities)

        # main loop (iteration)
        for self.it in tqdm.tqdm(range(0, self.iterations), ncols=100):

            # community check and event generation
            comp = nx.number_connected_components(self.graph)
            if self.it > 2*self.avg_deg and comp <= len(self.communities):
                if self.__test_communities():
                    self.node_to_com, self.communities, self.communities_involved = self.__generate_event(simplified)

            # node removal
            ar = random.random()
            if ar < self.del_node:
                self.__remove_node()

            # node addition
            ar = random.random()
            if ar < self.new_node:
                self.__add_node()

            self.out_interactions.flush()

            # get nodes within selected communities
            if len(self.communities_involved) == 0:
                nodes = self.graph.nodes()
            else:
                nodes = []
                for ci in self.communities_involved:
                    nodes.extend(self.communities[ci])
                nodes = set(nodes)
            random.shuffle(list(nodes))

            # inner loop (nodes)
            for n in nodes:

                # discard deleted nodes
                if self.node_to_com[n] == -1:
                    continue

                # check for decayed edges
                nn = nx.all_neighbors(self.graph, n)

                removal = []
                for n1 in nn:
                    delay = self.graph.get_edge_data(n, n1)['d']
                    if delay == self.it:
                        removal.append(n1)

                # removal phase
                for n1 in removal:
                    r = random.random()

                    # edge renewal phase
                    # check for intra/inter renewal thresholds
                    if r <= self.renewal and self.node_to_com[n1] == self.node_to_com[n]\
                            or r > self.renewal and self.node_to_com[n1] != self.node_to_com[n]:

                        # Exponential decay
                        timeout = (self.it + 1) + int(random.expovariate(self.lambdad))
                        self.graph.edge[n][n1]["d"] = timeout

                    else:
                        # edge to be removed
                        self.out_interactions.write("%s\t%s\t-\t%s\t%s\n" % (self.it, self.count, n, n1))
                        self.graph.remove_edge(n, n1)

                if self.graph.degree(n) >= self.exp_node_degs[n]:
                    continue

                # decide if the node is active during this iteration
                action = random.random()

                # the node has not yet reached it expected degree and it acts in this round
                if self.graph.degree(n) < self.exp_node_degs[n] and (action <= self.paction or self.it == 0):

                    com_nodes = set(self.communities[self.node_to_com[n]])

                    # probability for intra/inter community edges
                    r = random.random()

                    # check if at least sigma% of the node link are within the community
                    s = self.graph.subgraph(self.communities[self.node_to_com[n]])
                    d = s.degree(n)

                    # Intra-community edge
                    if d < len(com_nodes) - 1 and r <= self.sigma:
                        n_neigh = set(s.neighbors(n))

                        random.shuffle(list(n_neigh))
                        target = None

                        # selecting target node
                        candidates = {j: (self.exp_node_degs[j] - self.graph.degree(j)) for j in s.nodes()
                                      if (self.exp_node_degs[j] - self.graph.degree(j)) > 0 and j != n}

                        if len(candidates) > 0:
                            try:
                                target = random.sample(candidates, 1)[0]
                            except:
                                continue

                        # Interaction Exponential decay
                        timeout = (self.it + 1) + int(random.expovariate(self.lambdad))

                        # Edge insertion
                        if target is not None and not self.graph.has_edge(n, target) and target != n:
                            self.graph.add_edge(n, target, {"d": timeout})
                            self.count += 1
                            self.out_interactions.write("%s\t%s\t+\t%s\t%s\n" % (self.it, self.count, n, target))
                        else:
                            continue

                    # inter-community edges
                    elif r > self.sigma and \
                            self.exp_node_degs[n]-d < (1-self.sigma) * len(s.nodes()):

                        # randomly identifying a target community
                        try:
                            cid = random.sample(set(self.communities.keys()) - {self.node_to_com[n]}, 1)[0]
                        except:
                            continue

                        s = self.graph.subgraph(self.communities[cid])

                        # check for available nodes within the identified community
                        candidates = {j: (self.exp_node_degs[j] - self.graph.degree(j)) for j in s.nodes()
                                      if (self.exp_node_degs[j] - self.graph.degree(j)) > 0 and j != n}

                        # PA selection on available community nodes
                        if len(candidates) > 0:
                            candidatesp = list(np.array(candidates.values(), dtype='float') / sum(candidates.values()))
                            target = np.random.choice(candidates.keys(), 1, candidatesp)[0]

                            if self.graph.has_node(target) and not self.graph.has_edge(n, target):

                                # Interaction exponential decay
                                timeout = (self.it + 1) + int(random.expovariate(self.lambdad))
                                self.graph.add_edge(n, target, {"d": timeout})
                                self.count += 1
                                self.out_interactions.write("%s\t%s\t+\t%s\t%s\n" % (self.it, self.count, n, target))

        self.__output_communities()
        self.out_events.write("%s\n\t%s\n" % (self.iterations, self.performed_community_action))
        self.out_interactions.flush()
        self.out_interactions.close()
        self.out_events.flush()
        self.out_events.close()
        return self.stable


def main():
    import argparse

    sys.stdout.write("-------------------------------------\n")
    sys.stdout.write("               {RDyn}                \n")
    sys.stdout.write("           Graph Generator      \n")
    sys.stdout.write("     Handling Community Dynamics  \n")
    sys.stdout.write("-------------------------------------\n")
    sys.stdout.write("Author: " + __author__ + "\n")
    sys.stdout.write("Email:  " + __email__ + "\n")
    sys.stdout.write("------------------------------------\n")

    parser = argparse.ArgumentParser()
    parser.add_argument('nodes', type=int, help='Number of nodes', default=1000)
    parser.add_argument('iterations', type=int, help='Number of iterations', default=1000)
    parser.add_argument('simplified', type=bool, help='Simplified execution', default=True)
    parser.add_argument('-d', '--avg_degree', type=int, help='Average node degree', default=15)
    parser.add_argument('-s', '--sigma', type=float, help='Sigma', default=0.7)
    parser.add_argument('-l', '--lbd', type=float, help='Lambda community size distribution', default=1)
    parser.add_argument('-a', '--alpha', type=int, help='Alpha degree distribution', default=2.5)
    parser.add_argument('-p', '--prob_action', type=float, help='Probability of node action', default=1)
    parser.add_argument('-r', '--prob_renewal', type=float, help='Probability of edge renewal', default=0.8)
    parser.add_argument('-q', '--quality_threshold', type=float, help='Conductance quality threshold', default=0.3)
    parser.add_argument('-n', '--new_nodes', type=float, help='Probability of node appearance', default=0)
    parser.add_argument('-j', '--delete_nodes', type=float, help='Probability of node vanishing', default=0)
    parser.add_argument('-e', '--max_events', type=int, help='Max number of community events for stable iteration', default=1)

    args = parser.parse_args()
    rdyn = RDyn(size=args.nodes, iterations=args.iterations, avg_deg=args.avg_degree,
                sigma=args.sigma, lambdad=args.lbd, alpha=args.alpha, paction=args.prob_action,
                prenewal=args.prob_renewal, quality_threshold=args.quality_threshold,
                new_node=args.new_nodes, del_node=args.delete_nodes, max_evts=args.max_events)
    rdyn.execute(simplified=args.simplified)
