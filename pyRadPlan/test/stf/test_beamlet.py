from pyRadPlan.stf import IonSpot, PhotonBixel
from pyRadPlan.stf import RangeShifter


def test_proton_beamlet_only_energy():
    proton_beamlet = IonSpot(energy=100)
    assert proton_beamlet.energy == 100
    assert proton_beamlet.num_particles_per_mu == 1.0e6
    assert proton_beamlet.min_mu == 0.0
    assert proton_beamlet.max_mu == float("inf")
    assert isinstance(proton_beamlet.range_shifter, RangeShifter)
    assert proton_beamlet.focus_ix == 0


def test_proton_beamlet_all():
    proton_beamlet = IonSpot(
        energy=100,
        num_particles_per_mu=1.0e7,
        min_mu=1.0,
        max_mu=10.0,
        range_shifter=RangeShifter(),
        focus_ix=1,
    )
    assert proton_beamlet.energy == 100
    assert proton_beamlet.num_particles_per_mu == 1.0e7
    assert proton_beamlet.min_mu == 1.0
    assert proton_beamlet.max_mu == 10.0
    assert isinstance(proton_beamlet.range_shifter, RangeShifter)
    assert proton_beamlet.focus_ix == 1


def test_photon_beamlet_only_energy():
    photon_beamlet = PhotonBixel(energy=6)
    assert photon_beamlet.energy == 6
    assert photon_beamlet.num_particles_per_mu == 1.0e6
    assert photon_beamlet.min_mu == 0.0
    assert photon_beamlet.max_mu == float("inf")
    assert photon_beamlet.relative_fluence == 1.0


def test_photon_beamlet_all():
    photon_beamlet = PhotonBixel(
        energy=6,
        num_particles_per_mu=1.0e7,
        min_mu=1.0,
        max_mu=10.0,
        relative_fluence=0.5,
    )
    assert photon_beamlet.energy == 6
    assert photon_beamlet.num_particles_per_mu == 1.0e7
    assert photon_beamlet.min_mu == 1.0
    assert photon_beamlet.max_mu == 10.0
    assert photon_beamlet.relative_fluence == 0.5
