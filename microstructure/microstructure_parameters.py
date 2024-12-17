import numpy as np
import openpnm as op
import porespy as ps

np.set_printoptions(precision=4)
np.random.seed(10)




def calculate_permeability(img, resolution,p1,p2):
    filled =ps.filters.fill_blind_pores(img,surface=False)
    thickness = 2
    if p1 == 'zmin':
        filled [:, :, :thickness] = True  # Front face
    elif p1== 'ymin':
        filled [:, :thickness, :] = True  # Front face
    elif p1 == 'xmin' :
        filled [:thickness, :, :] = True  # Front face
    else:
        print ('Cluster Connector plance not found ERROR!!!!')
    snow = ps.networks.snow2(filled, voxel_size= resolution)
    pn = op.io.network_from_porespy(snow.network)

    # Something related to the recent update of the library (ignore it)
    pn['pore.diameter'] = pn['pore.equivalent_diameter']
    pn['throat.diameter'] = pn['throat.inscribed_diameter']
    pn['throat.spacing'] = pn['throat.total_length']

    # Adding model
    pn.add_model(propname='throat.hydraulic_size_factors',
                 model=op.models.geometry.hydraulic_size_factors.pyramids_and_cuboids)
    pn.add_model(propname='throat.diffusive_size_factors',
                 model=op.models.geometry.diffusive_size_factors.pyramids_and_cuboids)
    pn.regenerate_models()

    #Health check
    h = op.utils.check_network_health(pn)
    op.topotools.trim(network=pn, pores=h['disconnected_pores'])
    h = op.utils.check_network_health(pn)

    # p1 , p2 should be from pn labels, use print(pn) to check labels
    phase = op.phase.Phase(network=pn)
    phase['pore.viscosity']=1.0
    phase.add_model_collection(op.models.collections.physics.basic)
    phase.regenerate_models()

    def compute_k(pn, p1,p2):
        inlet = pn.pores(p1)
        outlet = pn.pores(p2)
        flow = op.algorithms.StokesFlow(network=pn, phase=phase)
        flow.set_value_BC(pores=inlet, values=1)
        flow.set_value_BC(pores=outlet, values=0)
        flow.run()
        phase.update(flow.soln)
        Q = flow.rate(pores=inlet, mode='group')[0]
        A = (img.shape[1] * img.shape[2])*(resolution**2) #A = op.topotools.get_domain_area(pn, inlets=inlet, outlets=outlet)
        L = img.shape[0]*resolution #L = op.topotools.get_domain_length(pn, inlets=inlet, outlets=outlet)
        k = Q * L / A # K = Q * L * mu / (A * Delta_P) # mu and Delta_P were assumed to be 1.
        return k

    #compute permeability on different directions
    k = compute_k(pn,p1,p2)

    return k


def calculate_tortuosity(img, resolution,p1,p2):
    filled =ps.filters.fill_blind_pores(img,surface=False)
    thickness = 2
    if p1 == 'zmin':
        filled [:, :, :thickness] = True  # Front face
    elif p1== 'ymin':
        filled [:, :thickness, :] = True  # Front face
    elif p1 == 'xmin' :
        filled [:thickness, :, :] = True  # Front face
    else:
        print ('Cluster Connector plance not found ERROR!!!!')
    snow = ps.networks.snow2(filled, voxel_size= resolution)
    
    pn = op.io.network_from_porespy(snow.network)

    # Something related to the recent update of the library (ignore it)
    pn['pore.diameter'] = pn['pore.equivalent_diameter']
    pn['throat.diameter'] = pn['throat.inscribed_diameter']
    pn['throat.spacing'] = pn['throat.total_length']

    # Adding model and geometry
    geo = op.models.collections.geometry.spheres_and_cylinders
    pn.add_model_collection(geo, domain='all')
    pn.regenerate_models()

    #Health check
    h = op.utils.check_network_health(pn)
    op.topotools.trim(network=pn, pores=h['disconnected_pores'])
    h = op.utils.check_network_health(pn)

    # p1 , p2 should be from pn labels, use print(pn) to check labels
    air = op.phase.Air(network=pn)

    phys = op.models.collections.physics.basic
    #del phys['throat.entry_pressure']
    air.add_model_collection(phys)
    air.regenerate_models()

    fd = op.algorithms.FickianDiffusion(network=pn, phase=air)


    def compute_tau(pn, p1,p2):
        inlet = pn.pores(p1)
        outlet = pn.pores(p2)
        C_in, C_out = [10, 5]
        fd.set_value_BC(pores=inlet, values=C_in)
        fd.set_value_BC(pores=outlet, values=C_out)

        fd.run()
        rate_inlet = fd.rate(pores=inlet)[0]
        #print(f'Molar flow rate: {rate_inlet:.5e} mol/s')

        A = (img.shape[1] * img.shape[2])*(resolution**2)
        L = img.shape[0]*resolution
        D_eff = rate_inlet * L / (A * (C_in - C_out))
        #print("{0:.6E}".format(D_eff))


        V_p = pn['pore.volume'].sum()
        V_t = pn['throat.volume'].sum()
        V_bulk = np.prod(img.shape)*(resolution**3)

        e = (V_p + V_t) / V_bulk
        #print('The porosity is: ', "{0:.6E}".format(e))


        D_AB = air['pore.diffusivity'][0]
        tau = e * D_AB / D_eff
        #print('The tortuosity is:', "{0:.6E}".format(tau))
        return tau

    #compute permeability on different directions
    tau = compute_tau(pn,p1,p2)

    return tau
