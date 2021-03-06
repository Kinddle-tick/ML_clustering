"""!

@brief Examples of usage and demonstration of abilities of ROCK algorithm in cluster analysis.

@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2020
@copyright BSD-3-Clause

"""

from pyclustering.cluster.rock import rock;

from pyclustering.samples.definitions import SIMPLE_SAMPLES;
from pyclustering.samples.definitions import FCPS_SAMPLES;

from pyclustering.utils import read_sample;
from pyclustering.utils import draw_clusters;
from pyclustering.utils import timedcall;

def template_clustering(path, radius, cluster_numbers, threshold, draw = True, ccore = True):
    sample = read_sample(path);
    
    rock_instance = rock(sample, radius, cluster_numbers, threshold, ccore);
    (ticks, result) = timedcall(rock_instance.process);
    
    clusters = rock_instance.get_clusters();
    
    print("Sample: ", path, "\t\tExecution time: ", ticks, "\n");
    
    if (draw == True):
        draw_clusters(sample, clusters);
    
def cluster_simple1():
    template_clustering(SIMPLE_SAMPLES.SAMPLE_SIMPLE1, 1, 2, 0.5);
    
def cluster_simple2():
    template_clustering(SIMPLE_SAMPLES.SAMPLE_SIMPLE2, 1, 3, 0.5);
    
def cluster_simple3():
    template_clustering(SIMPLE_SAMPLES.SAMPLE_SIMPLE3, 1, 4, 0.5);
    
def cluster_simple4():
    template_clustering(SIMPLE_SAMPLES.SAMPLE_SIMPLE4, 1, 5, 0.5);

def cluster_simple5():
    template_clustering(SIMPLE_SAMPLES.SAMPLE_SIMPLE5, 1, 4, 0.5);
    
def cluster_elongate():
    template_clustering(SIMPLE_SAMPLES.SAMPLE_ELONGATE, 1, 2, 0.5); 
    
def cluster_lsun():
    template_clustering(FCPS_SAMPLES.SAMPLE_LSUN, 1, 3, 0.5);       

def cluster_target():
    template_clustering(FCPS_SAMPLES.SAMPLE_TARGET, 1.2, 6, 0.2);     

def cluster_two_diamonds():
    template_clustering(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS, 0.2, 2, 0.2);  

def cluster_wing_nut():
    template_clustering(FCPS_SAMPLES.SAMPLE_WING_NUT, 0.3, 2, 0.2); 
    
def cluster_chainlink():
    template_clustering(FCPS_SAMPLES.SAMPLE_CHAINLINK, 0.6, 2, 0.2);     
    
def cluster_hepta():
    template_clustering(FCPS_SAMPLES.SAMPLE_HEPTA, 1.2, 7, 0.2); 
    
def cluster_tetra():
    template_clustering(FCPS_SAMPLES.SAMPLE_TETRA, 0.5, 4, 0.2);  
    
def experiment_execution_time(ccore):
    template_clustering(SIMPLE_SAMPLES.SAMPLE_SIMPLE1, 1, 2, 0.5, False, ccore);
    template_clustering(SIMPLE_SAMPLES.SAMPLE_SIMPLE2, 1, 3, 0.5, False, ccore);
    template_clustering(SIMPLE_SAMPLES.SAMPLE_SIMPLE3, 1, 4, 0.5, False, ccore);
    template_clustering(SIMPLE_SAMPLES.SAMPLE_SIMPLE4, 1, 5, 0.5, False, ccore);
    template_clustering(SIMPLE_SAMPLES.SAMPLE_ELONGATE, 1, 2, 0.5, False, ccore);
    template_clustering(FCPS_SAMPLES.SAMPLE_LSUN, 1, 3, 0.5, True, ccore);
    template_clustering(FCPS_SAMPLES.SAMPLE_TARGET, 1.2, 6, 0.2, True, ccore);
    template_clustering(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS, 0.2, 2, 0.2, True, ccore);
    template_clustering(FCPS_SAMPLES.SAMPLE_WING_NUT, 0.3, 2, 0.2, True, ccore);
    template_clustering(FCPS_SAMPLES.SAMPLE_CHAINLINK, 0.6, 2, 0.2, True, ccore);
    template_clustering(FCPS_SAMPLES.SAMPLE_HEPTA, 1.2, 7, 0.2, True, ccore);
    template_clustering(FCPS_SAMPLES.SAMPLE_TETRA, 0.5, 4, 0.2, True, ccore);
    template_clustering(FCPS_SAMPLES.SAMPLE_ATOM, 15, 2, 0.2, True, ccore)


cluster_simple1();
cluster_simple2();
cluster_simple3();
cluster_simple4();
cluster_simple5();
cluster_elongate();
cluster_lsun();
cluster_target();
cluster_two_diamonds();
cluster_wing_nut();
cluster_chainlink();
cluster_hepta();
cluster_tetra();
 
 
experiment_execution_time(False);   # Slow Thecode - python
# experiment_execution_time(True);    # Fast Thecode - C++ CCORE