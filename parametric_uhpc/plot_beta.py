import numpy as np
import matplotlib.pyplot as plt

def plotBetaVsMandK(beta_all, Envelope, beta_z1, beta_z2, beta_z3, beta_z4,
                    beta111=None, M111=None, k111=None, beta211=None, M211=None, k211=None, beta212=None, M212=None, k212=None,
                    beta221=None, M221=None, k221=None, beta222=None, M222=None, k222=None,
                    beta311=None, M311=None, k311=None, beta312=None, M312=None, k312=None,
                    beta321=None, M321=None, k321=None, beta322=None, M322=None, k322=None,
                    beta411=None, M411=None, k411=None, beta412=None, M412=None, k412=None,
                    beta421=None, M421=None, k421=None, beta422=None, M422=None, k422=None,
                    beta4222=None, M4222=None, k4222=None):

    # Determine marker spacing
    markerSpacing = int(len(beta_all) * 0.1)  # Change this value to adjust spacing

    # Plot for Beta vs M
    plt.figure()
    plt.title('Beta vs M for Different Zones')
    plt.xlabel('Beta')
    plt.ylabel('M')
    
    # Check and plot each M variable
    if M111 is not None:
        plt.plot(beta111, M111, '-o', label='zone111', linewidth=1, color='k', markevery=markerSpacing)
    if M211 is not None:
        plt.plot(beta211, M211, '-o', label='zone211', linewidth=1, color='m', markevery=markerSpacing)
    if M212 is not None:
        plt.plot(beta212, M212, '-^', label='zone212', linewidth=1, color='m', markevery=markerSpacing)
    if M221 is not None:
        plt.plot(beta221, M221, '-s', label='zone221', linewidth=1, color='m', markevery=markerSpacing)
    if M222 is not None:
        plt.plot(beta222, M222, '-d', label='zone222', linewidth=1, color='m', markevery=markerSpacing)
    if M311 is not None:
        plt.plot(beta311, M311, '-o', label='zone311', linewidth=1, color='b', markevery=markerSpacing)
    if M312 is not None:
        plt.plot(beta312, M312, '-^', label='zone312', linewidth=1, color='b', markevery=markerSpacing)
    if M321 is not None:
        plt.plot(beta321, M321, '-s', label='zone321', linewidth=1, color='b', markevery=markerSpacing)
    if M322 is not None:
        plt.plot(beta322, M322, '-d', label='zone322', linewidth=1, color='b', markevery=markerSpacing)
    if M411 is not None:
        plt.plot(beta411, M411, '-o', label='zone411', linewidth=1, color='r', markevery=markerSpacing)
    if M412 is not None:
        plt.plot(beta412, M412, '-^', label='zone412', linewidth=1, color='r', markevery=markerSpacing)
    if M421 is not None:
        plt.plot(beta421, M421, '-s', label='zone421', linewidth=1, color='r', markevery=markerSpacing)
    if M422 is not None:
        plt.plot(beta422, M422, '-d', label='zone422', linewidth=1, color='r', markevery=markerSpacing)
    if M4222 is not None:
        plt.plot(beta4222, M4222, '-*', label='zone4222', linewidth=1, color='g', markevery=markerSpacing)

    # Add the envelope plot
    beta_plot = np.concatenate((beta_z1, beta_z2, beta_z3, beta_z4))
    plt.plot(beta_plot, Envelope[:len(beta_plot), 1], label='Envelope', linewidth=5, color='black', alpha=0.5)
    plt.ylim([0, max(Envelope[:len(beta_plot), 1]) * 1.2])
    plt.legend()
    plt.show()

    # Plot for Beta vs k
    plt.figure()
    plt.title('Beta vs k for Different Zones')
    plt.xlabel('Beta')
    plt.ylabel('k')

    # Check and plot each k variable
    if k111 is not None:
        plt.plot(beta111, k111, '-o', label='zone111', linewidth=1, color='k', markevery=markerSpacing)
    if k211 is not None:
        plt.plot(beta211, k211, '-o', label='zone211', linewidth=1, color='m', markevery=markerSpacing)
    if k212 is not None:
        plt.plot(beta212, k212, '-^', label='zone212', linewidth=1, color='m', markevery=markerSpacing)
    if k221 is not None:
        plt.plot(beta221, k221, '-s', label='zone221', linewidth=1, color='m', markevery=markerSpacing)
    if k222 is not None:
        plt.plot(beta222, k222, '-d', label='zone222', linewidth=1, color='m', markevery=markerSpacing)
    if k311 is not None:
        plt.plot(beta311, k311, '-o', label='zone311', linewidth=1, color='b', markevery=markerSpacing)
    if k312 is not None:
        plt.plot(beta312, k312, '-^', label='zone312', linewidth=1, color='b', markevery=markerSpacing)
    if k321 is not None:
        plt.plot(beta321, k321, '-s', label='zone321', linewidth=1, color='b', markevery=markerSpacing)
    if k322 is not None:
        plt.plot(beta322, k322, '-d', label='zone322', linewidth=1, color='b', markevery=markerSpacing)
    if k411 is not None:
        plt.plot(beta411, k411, '-o', label='zone411', linewidth=1, color='r', markevery=markerSpacing)
    if k412 is not None:
        plt.plot(beta412, k412, '-^', label='zone412', linewidth=1, color='r', markevery=markerSpacing)
    if k421 is not None:
        plt.plot(beta421, k421, '-s', label='zone421', linewidth=1, color='r', markevery=markerSpacing)
    if k422 is not None:
        plt.plot(beta422, k422, '-d', label='zone422', linewidth=1, color='r', markevery=markerSpacing)
    if k4222 is not None:
        plt.plot(beta4222, k4222, '-*', label='zone4222', linewidth=1, color='g', markevery=markerSpacing)

    plt.legend()
    plt.show()
