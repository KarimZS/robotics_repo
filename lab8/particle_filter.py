from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy
import sys

def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments: 
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = []
    for particle in particles:
        dx,dy,dh = add_odometry_noise(odom,ODOM_HEAD_SIGMA,ODOM_TRANS_SIGMA)#add noise to the reading
        rdx,rdy=rotate_point(dx,dy,particle.h)#rotate
        newParticle = Particle(particle.x+rdx,particle.y+rdy,particle.h+dh)#create a new particle based on the movement from the noisy odometery
        motion_particles.append(newParticle)
    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments: 
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information, 
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    measured_particles = []
    particle_weights = []#weight for each particle
    sum_weights=0#sum of weights used to normalize later
    for particle in particles:
        prob = 0
        if grid.is_free(particle.x,particle.y):#particle in the grid
            particle_markers = particle.read_markers(grid)
            if len(measured_marker_list)==0:#there are no markers measured
                if len(particle_markers)==0:#particle also sees no markers
                    prob=1#max probability
                else:#particle sees markers, so not right spot
                    prob=0
            else:#robot sees marker
                for particle_marker in particle_markers:
                    particle_marker = add_marker_measurement_noise(particle_marker, MARKER_TRANS_SIGMA,MARKER_ROT_SIGMA)#add marker measurement noise
                    closest_marker = None
                    closest_marker_distance = sys.maxsize#distance between seen marker and robot seen markers
                    for measured_marker in measured_marker_list:#find the closest matching marker from seen and point
                        distance = grid_distance(measured_marker[0],measured_marker[1],particle_marker[0],particle_marker[1])
                        if distance < closest_marker_distance:
                            closest_marker = measured_marker
                            closest_marker_distance = distance
                    heading_diff = abs(diff_heading_deg(closest_marker[2],particle_marker[2]))#find heading diff between the 2 markers
                    prob=min(((1/closest_marker_distance)+1/(heading_diff))+prob,1)#prob is sum of marker closeness (if match then 1) with a max of 1
        else:#particle not in grid so prob=0
            prob=0
        particle_weights.append(prob)#add particle weight to list
        sum_weights+=prob#add to sum

    particle_weights = [weight / sum_weights for weight in particle_weights]#normalize the weights to sum to 1
    particle_count = len(particles)
    random_particles_count = int(particle_count*.01)#add 1% random
    resampling_size = particle_count - random_particles_count
    measured_particles = numpy.random.choice(particles, size = resampling_size, p = particle_weights).tolist()#choose randomly based on weight
    measured_particles+=Particle.create_random(random_particles_count, grid)#add random particles
    return measured_particles


