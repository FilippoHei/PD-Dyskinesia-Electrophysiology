import numpy as np
import mat73
import pyvista as pv
from scipy.spatial import ConvexHull, Delaunay

class Distal_Atlas:
    
    def __init__(self, threshold):
        
        self.bg_threshold  = threshold   # the probabilistic threshold for DISTAL Atlas
        self.atlas         = mat73.loadmat('atlases/DISTAL Atlas (Medium)/atlas_index.mat')
        self.stn_index     = 0           # index of STN
        self.stn_m_index   = 1           # index of STN motor area
        self.stn_a_index   = 2           # index of STN associative area
        self.stn_l_index   = 3           # index of STN limbic area
        self.gpi_index     = 4           # index of GPi
        self.rn_index      = 13          # index of RN
        self.gpe_index     = 14          # index of GPe
        self.tha_index     = 15          # index of Thalamus
        self.va_index      = 16          # index of VA
        self.vim_index     = 17          # index of VIM
        self.cm_index      = 18          # index of CM
        self.point_perct   = 0.2         # the percentage of points will be used for meshing for big basal ganglia nuclei
        self.basal_ganglia = self.__get_nuclei_definitions()
        self.mesh          = self.__acquire_bg_meshes()
        
        self.mesh_style    = 'surface'
        self.show_edges    = True
        
        self.mesh_graps    = {}

        # STN and its functional area definitions
        self.STN                         = {}
        self.STN["right"]                = {}
        self.STN["right"]["motor"]       = self.get_nucleus_3D_definition(nucleus="stn_m", hemisphere="right")
        self.STN["right"]["associative"] = self.get_nucleus_3D_definition(nucleus="stn_a", hemisphere="right")
        self.STN["right"]["limbic"]      = self.get_nucleus_3D_definition(nucleus="stn_l", hemisphere="right")
        self.STN["left"]                 = {}
        self.STN["left"]["motor"]        = self.get_nucleus_3D_definition(nucleus="stn_m", hemisphere="left")
        self.STN["left"]["associative"]  = self.get_nucleus_3D_definition(nucleus="stn_a", hemisphere="left")
        self.STN["left"]["limbic"]       = self.get_nucleus_3D_definition(nucleus="stn_l", hemisphere="left")
    
    def __get_nuclei_definitions(self):
        basal_ganglia              = {}
        basal_ganglia["dx"]        = {}
        basal_ganglia["sx"]        = {}
        
        basal_ganglia["dx"]["stn"]   = self.atlas["atlases"]["XYZ"][self.stn_index][0]["mm"][self.atlas["atlases"]["XYZ"][self.stn_index][0]["val"] >= self.bg_threshold]
        basal_ganglia["sx"]["stn"]   = self.atlas["atlases"]["XYZ"][self.stn_index][1]["mm"][self.atlas["atlases"]["XYZ"][self.stn_index][1]["val"] >= self.bg_threshold]
        basal_ganglia["dx"]["stn_m"] = self.atlas["atlases"]["XYZ"][self.stn_m_index][0]["mm"][self.atlas["atlases"]["XYZ"][self.stn_m_index][0]["val"] >= self.bg_threshold]
        basal_ganglia["sx"]["stn_m"] = self.atlas["atlases"]["XYZ"][self.stn_m_index][1]["mm"][self.atlas["atlases"]["XYZ"][self.stn_m_index][1]["val"] >= self.bg_threshold]
        basal_ganglia["dx"]["stn_a"] = self.atlas["atlases"]["XYZ"][self.stn_a_index][0]["mm"][self.atlas["atlases"]["XYZ"][self.stn_a_index][0]["val"] >= self.bg_threshold]
        basal_ganglia["sx"]["stn_a"] = self.atlas["atlases"]["XYZ"][self.stn_a_index][1]["mm"][self.atlas["atlases"]["XYZ"][self.stn_a_index][1]["val"] >= self.bg_threshold]
        basal_ganglia["dx"]["stn_l"] = self.atlas["atlases"]["XYZ"][self.stn_l_index][0]["mm"][self.atlas["atlases"]["XYZ"][self.stn_l_index][0]["val"] >= self.bg_threshold]
        basal_ganglia["sx"]["stn_l"] = self.atlas["atlases"]["XYZ"][self.stn_l_index][1]["mm"][self.atlas["atlases"]["XYZ"][self.stn_l_index][1]["val"] >= self.bg_threshold]
        basal_ganglia["dx"]["gpi"]   = self.atlas["atlases"]["XYZ"][self.gpi_index][0]["mm"][self.atlas["atlases"]["XYZ"][self.gpi_index][0]["val"] >= self.bg_threshold]
        basal_ganglia["sx"]["gpi"]   = self.atlas["atlases"]["XYZ"][self.gpi_index][1]["mm"][self.atlas["atlases"]["XYZ"][self.gpi_index][1]["val"] >= self.bg_threshold]
        basal_ganglia["dx"]["gpe"]   = self.atlas["atlases"]["XYZ"][self.gpe_index][0]["mm"][self.atlas["atlases"]["XYZ"][self.gpe_index][0]["val"] >= self.bg_threshold]
        basal_ganglia["sx"]["gpe"]   = self.atlas["atlases"]["XYZ"][self.gpe_index][1]["mm"][self.atlas["atlases"]["XYZ"][self.gpe_index][1]["val"] >= self.bg_threshold]
        basal_ganglia["dx"]["tha"]   = self.atlas["atlases"]["XYZ"][self.tha_index][0]["mm"][self.atlas["atlases"]["XYZ"][self.tha_index][0]["val"] >= self.bg_threshold]
        basal_ganglia["sx"]["tha"]   = self.atlas["atlases"]["XYZ"][self.tha_index][1]["mm"][self.atlas["atlases"]["XYZ"][self.tha_index][1]["val"] >= self.bg_threshold]
        basal_ganglia["dx"]["rn"]    = self.atlas["atlases"]["XYZ"][self.rn_index][0]["mm"][self.atlas["atlases"]["XYZ"][self.rn_index][0]["val"]   >= self.bg_threshold]
        basal_ganglia["sx"]["rn"]    = self.atlas["atlases"]["XYZ"][self.rn_index][1]["mm"][self.atlas["atlases"]["XYZ"][self.rn_index][1]["val"]   >= self.bg_threshold]
        basal_ganglia["dx"]["va"]    = self.atlas["atlases"]["XYZ"][self.va_index][0]["mm"][self.atlas["atlases"]["XYZ"][self.va_index][0]["val"]   >= self.bg_threshold]
        basal_ganglia["sx"]["va"]    = self.atlas["atlases"]["XYZ"][self.va_index][1]["mm"][self.atlas["atlases"]["XYZ"][self.va_index][1]["val"]   >= self.bg_threshold]
        basal_ganglia["dx"]["vim"]   = self.atlas["atlases"]["XYZ"][self.vim_index][0]["mm"][self.atlas["atlases"]["XYZ"][self.vim_index][0]["val"] >= self.bg_threshold]
        basal_ganglia["sx"]["vim"]   = self.atlas["atlases"]["XYZ"][self.vim_index][1]["mm"][self.atlas["atlases"]["XYZ"][self.vim_index][1]["val"] >= self.bg_threshold]
        basal_ganglia["dx"]["cm"]    = self.atlas["atlases"]["XYZ"][self.cm_index][0]["mm"][self.atlas["atlases"]["XYZ"][self.cm_index][0]["val"]   >= self.bg_threshold]
        basal_ganglia["sx"]["cm"]    = self.atlas["atlases"]["XYZ"][self.cm_index][1]["mm"][self.atlas["atlases"]["XYZ"][self.cm_index][1]["val"]   >= self.bg_threshold]
        
        return basal_ganglia
    
    
    def __acquire_bg_meshes(self):
        mesh           = {}
        mesh["dx"]     = {}
        mesh["sx"]     = {}
        mesh["colors"] = {}
        mesh["labels"] = {}
        
        if(self.bg_threshold == 0.5):
            stn_dx_mesh = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/stn_dx.stl")
            stn_sx_mesh = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/stn_sx.stl")
            gpi_dx_mesh = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/gpi_dx.stl")
            gpi_sx_mesh = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/gpi_sx.stl")
            gpe_dx_mesh = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/gpe_dx.stl")
            gpe_sx_mesh = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/gpe_sx.stl")
            tha_dx_mesh = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/tha_dx.stl")
            tha_sx_mesh = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/tha_sx.stl")
            rn_dx_mesh  = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/rn_dx.stl")
            rn_sx_mesh  = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/rn_sx.stl")
            van_dx_mesh = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/va_dx.stl")
            van_sx_mesh = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/va_sx.stl")
            vim_dx_mesh = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/vim_dx.stl")
            vim_sx_mesh = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/vim_sx.stl")
            cm_dx_mesh  = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/cm_dx.stl")
            cm_sx_mesh  = pv.read("atlases/DISTAL Atlas (Medium)/3D Models/threshold_0.5/cm_sx.stl")

            mesh["dx"]["stn"] = stn_dx_mesh
            mesh["dx"]["gpi"] = gpi_dx_mesh
            mesh["dx"]["gpe"] = gpe_dx_mesh
            mesh["dx"]["tha"] = tha_dx_mesh
            mesh["dx"]["rn"]  = rn_dx_mesh
            mesh["dx"]["van"] = van_dx_mesh
            mesh["dx"]["vim"] = vim_dx_mesh
            mesh["dx"]["cm"]  = cm_dx_mesh
            mesh["sx"]["stn"] = stn_sx_mesh
            mesh["sx"]["gpi"] = gpi_sx_mesh
            mesh["sx"]["gpe"] = gpe_sx_mesh
            mesh["sx"]["tha"] = tha_sx_mesh
            mesh["sx"]["rn"]  = rn_sx_mesh
            mesh["sx"]["van"] = van_sx_mesh
            mesh["sx"]["vim"] = vim_sx_mesh
            mesh["sx"]["cm"]  = cm_sx_mesh

        mesh["colors"]["gpe"] = "lightseagreen"
        mesh["colors"]["gpi"] = "lightsalmon"
        mesh["colors"]["stn"] = "firebrick"
        mesh["colors"]["tha"] = "springgreen"
        mesh["colors"]["rn"]  = "gold"
        mesh["colors"]["van"] = "navy"
        mesh["colors"]["vim"] = "deeppink"
        mesh["colors"]["cm"]  = "limegreen"
        
        mesh["labels"]["gpe"]  = "Globus Pallidus Externa"
        mesh["labels"]["gpi"]  = "Globus Pallidus Interna"
        mesh["labels"]["stn"]  = "Subthalamic Nucleus"
        mesh["labels"]["tha"]  = "Thalamus"
        mesh["labels"]["rn"]   = "Red Nucleus"
        mesh["labels"]["van"]  = "Ventral Anterior Nucleus"
        mesh["labels"]["vim"]  = "Ventral Intermediate Nucleus"
        mesh["labels"]["cm"]   = "Centromedial Nucleus"
        
        return mesh
    
    
    def plot_nuclei(self, plotter, hemisphere, nuclei_visibitity, opacity):
        
        for nucleus in list(nuclei_visibitity.keys()):
            
            if(hemisphere=="both"):
                plotter.disable_depth_peeling()
                if(nuclei_visibitity[nucleus] == True):
                    plotter.add_mesh(self.mesh["dx"][nucleus], color=self.mesh["colors"][nucleus], opacity = opacity, name=(nucleus+"-dx"), 
                                     style=self.mesh_style, label=self.mesh["labels"][nucleus]) 
                    plotter.add_mesh(self.mesh["sx"][nucleus], color=self.mesh["colors"][nucleus], opacity = opacity, name=(nucleus+"-sx"),
                                     style=self.mesh_style)
                else:
                    plotter.add_mesh(self.mesh["dx"][nucleus], color="darkgray", opacity = opacity, name=(nucleus+"-dx"), style=self.mesh_style) 
                    plotter.add_mesh(self.mesh["sx"][nucleus], color="darkgray", opacity = opacity, name=(nucleus+"-sx"), style=self.mesh_style)
            elif(hemisphere=="right"):
                if(nuclei_visibitity[nucleus] == True):
                    plotter.add_mesh(self.mesh["dx"][nucleus], color=self.mesh["colors"][nucleus], opacity = opacity, name=(nucleus+"-dx"), 
                                     style=self.mesh_style, label=self.mesh["labels"][nucleus])                     
                else:
                    plotter.add_mesh(self.mesh["dx"][nucleus], color="darkgray", opacity = opacity, name=(nucleus+"-dx"), style=self.mesh_style)
                    plotter.add_mesh(self.mesh["sx"][nucleus], color=self.mesh["colors"][nucleus], opacity = 0, name=(nucleus+"-sx"), style=self.mesh_style)
            else:
                if(nuclei_visibitity[nucleus] == True):
                    plotter.add_mesh(self.mesh["sx"][nucleus], color=self.mesh["colors"][nucleus], opacity = opacity, name=(nucleus+"-sx"),
                                     style=self.mesh_style, label=self.mesh["labels"][nucleus])                     
                else:
                    plotter.add_mesh(self.mesh["sx"][nucleus], color="darkgray", opacity = opacity, name=(nucleus+"-sx"), style=self.mesh_style)
                    plotter.add_mesh(self.mesh["dx"][nucleus], color="darkgray", opacity = 0, name=(nucleus+"-dx"), style=self.mesh_style) 

        plotter.add_legend(face=None, size=(0.15, 0.15)) 
        #plotter.camera_position = 'xz'
        #plotter.camera.zoom(1.5)
        return plotter
    
    def plot_nuclei2(self, plotter, hemisphere, nuclei_visibitity, opacity):
        
        for nucleus in list(nuclei_visibitity.keys()):
            
            if(hemisphere=="both"):
                plotter.disable_depth_peeling()
                if(nuclei_visibitity[nucleus] == True):
                    plotter.add_mesh(self.mesh["dx"][nucleus], color=self.mesh["colors"][nucleus], opacity = opacity[nucleus], name=(nucleus+"-dx"), 
                                     style=self.mesh_style, label=self.mesh["labels"][nucleus]) 
                    plotter.add_mesh(self.mesh["sx"][nucleus], color=self.mesh["colors"][nucleus], opacity = opacity[nucleus], name=(nucleus+"-sx"),
                                     style=self.mesh_style)
                else:
                    plotter.add_mesh(self.mesh["dx"][nucleus], color="darkgray", opacity = opacity[nucleus], name=(nucleus+"-dx"), style=self.mesh_style) 
                    plotter.add_mesh(self.mesh["sx"][nucleus], color="darkgray", opacity = opacity[nucleus], name=(nucleus+"-sx"), style=self.mesh_style)
            elif(hemisphere=="right"):
                if(nuclei_visibitity[nucleus] == True):
                    plotter.add_mesh(self.mesh["dx"][nucleus], color=self.mesh["colors"][nucleus], opacity = opacity[nucleus], name=(nucleus+"-dx"), 
                                     style=self.mesh_style, label=self.mesh["labels"][nucleus])                     
                else:
                    plotter.add_mesh(self.mesh["dx"][nucleus], color="darkgray", opacity = opacity[nucleus], name=(nucleus+"-dx"), style=self.mesh_style)
                    plotter.add_mesh(self.mesh["sx"][nucleus], color=self.mesh["colors"][nucleus], opacity = 0, name=(nucleus+"-sx"), style=self.mesh_style)
            else:
                if(nuclei_visibitity[nucleus] == True):
                    plotter.add_mesh(self.mesh["sx"][nucleus], color=self.mesh["colors"][nucleus], opacity = opacity[nucleus], name=(nucleus+"-sx"),
                                     style=self.mesh_style, label=self.mesh["labels"][nucleus])                     
                else:
                    plotter.add_mesh(self.mesh["sx"][nucleus], color="darkgray", opacity = opacity[nucleus], name=(nucleus+"-sx"), style=self.mesh_style)
                    plotter.add_mesh(self.mesh["dx"][nucleus], color="darkgray", opacity = 0, name=(nucleus+"-dx"), style=self.mesh_style) 

        plotter.add_legend(face=None, size=(0.15, 0.15)) 
        return plotter
    
    def select_deselect(self, plotter, hemisphere, nucleus, state, opacity_target, opacity_nontarget):
        
        if(hemisphere=="both"):
            if(state==True):
                plotter.add_mesh(self.mesh["dx"][nucleus], color=self.mesh["colors"][nucleus], opacity = opacity_target, name=(nucleus+"-dx"), label=self.mesh["labels"][nucleus]) 
                plotter.add_mesh(self.mesh["sx"][nucleus], color=self.mesh["colors"][nucleus], opacity = opacity_target, name=(nucleus+"-sx"))
            else:
                plotter.add_mesh(self.mesh["dx"][nucleus], color="darkgray", opacity = opacity_nontarget, name=(nucleus+"-dx")) 
                plotter.add_mesh(self.mesh["sx"][nucleus], color="darkgray", opacity = opacity_nontarget, name=(nucleus+"-sx"))
        elif(hemisphere=="right"):
            if(state==True):
                plotter.add_mesh(self.mesh["dx"][nucleus], color=self.mesh["colors"][nucleus], opacity = opacity_target, name=(nucleus+"-dx"), label=self.mesh["labels"][nucleus]) 
            else:
                plotter.add_mesh(self.mesh["dx"][nucleus], color="darkgray", opacity = opacity_nontarget, name=(nucleus+"-dx"))
            plotter.add_mesh(self.mesh["sx"][nucleus], color=self.mesh["colors"][nucleus], opacity = 0, name=(nucleus+"-sx"))
        else:
            if(state==True):
                plotter.add_mesh(self.mesh["sx"][nucleus], color=self.mesh["colors"][nucleus], opacity = opacity_target, name=(nucleus+"-sx"), label=self.mesh["labels"][nucleus]) 
            else:
                plotter.add_mesh(self.mesh["sx"][nucleus], color="darkgray", opacity = opacity_nontarget, name=(nucleus+"-sx"))  
            plotter.add_mesh(self.mesh["dx"][nucleus], color="darkgray", opacity = 0, name=(nucleus+"-dx"))
            
            

        plotter.add_legend(face=None, size=(0.15, 0.15)) 
        #plotter.camera_position = 'xz'
        #plotter.camera.zoom(1.5)
        return plotter
    
    def show_neurons(self, plotter, neuron_locations, state):
        if(state==True):
            plotter.add_points(neuron_locations, point_size=4, name="neurons", opacity=0.99, color="w")
        else:
            plotter.add_points(neuron_locations, point_size=4, name="neurons", opacity=0, color="w")
        return plotter
    
    def get_nucleus_3D_definition(self, nucleus, hemisphere):
        
        hemisphere_code = "dx" if(hemisphere == "right") else "sx"
        return self.basal_ganglia[hemisphere_code][nucleus]

    def nuclei_in_out(self, nuclei, lead_coordinates):
    
        nuclei_position = []
        
        for coordinate in lead_coordinates:

            # apply convex hull algorithm to define if given coordinate list resides within the nucleus or not
            hull     = ConvexHull(nuclei)
            new_hull = ConvexHull(np.concatenate((nuclei, [coordinate])))
            nuclei_position.append(np.array_equal(new_hull.vertices, hull.vertices) == True)
        
        return nuclei_position

    def check_position_in_nucleus(self, nucleus, hemisphere, coordinates):
        
        if(self.nuclei_in_out(nucleus[hemisphere]["motor"], coordinates)[0]==True):
            return "motor"
        
        elif(self.nuclei_in_out(nucleus[hemisphere]["associative"], coordinates)[0]==True):
            return "associative"
       
        elif(self.nuclei_in_out(nucleus[hemisphere]["limbic"], coordinates)[0]==True):
            return "limbic"
        
        else:
            return "outside"
    