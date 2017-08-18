import unittest
import shutil
from rdyn.alg.RDyn import RDyn


class RDynTestCase(unittest.TestCase):

    def test_rdyn_simplified(self):

        rdb = RDyn(size=1000, iterations=100)
        rdb.execute(simplified=True)

        rdb = RDyn(size=100, iterations=10)
        try:
            rdb.execute(simplified=True)
        except:
            pass

        rdb = RDyn(size=1000, iterations=100, new_node=0.1, del_node=0.1, max_evts=2, paction=0.8)
        rdb.execute(simplified=False)

        shutil.rmtree("results")

if __name__ == '__main__':
    unittest.main()
