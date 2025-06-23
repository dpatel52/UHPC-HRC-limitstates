import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_doubly_reinforced_beam(
    length, height, width, cover, topDiameter, topCount, botDiameter, botCount, mu, ax=None, sideview=False,
    loadingType=None, loadSpacing=None
):
    """
    Draw a doubly reinforced concrete beam cross‐section with randomly oriented fibers.
    If sideview=True, also plot a 2D side view as a subplot.
    """
    
    if sideview:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        ax = axs[0]
        show_fig = True
    else:
        if ax is None:
            fig, ax = plt.subplots()
            show_fig = True
        else:
            show_fig = False

    ax.clear()
    ax.set_aspect('equal')
    
    # Define colors
    concreteColor = (0.7, 0.7, 0.7)  # Gray
    rebarColor    = (0.8, 0.2, 0.2)  # Reddish
    fiberColor    = (0, 0, 0)         # Black
    
    # Draw concrete cross section
    concrete_rect = patches.Rectangle((0, 0), width, height,
                                      facecolor=concreteColor, edgecolor='k')
    ax.add_patch(concrete_rect)
    
    # Calculate rebar center positions
    topRebarY = height - cover
    botRebarY = cover

    # Draw random fibers if applicable.
    # Determine number and length of fibers.
    if width > 74 or height > 74:
        numFibers = int(mu * 300)
        fiberLength = 10  # e.g., 10 mm (approx. 1 inch)
    else:
        numFibers = int(mu * (height * width) * 1.1)
        fiberLength = 1   # e.g., 1 unit
    
    fiberWidth = 0.2  # Thin fibers
    
    for j in range(numFibers):
        # Generate a random angle in radians (0 to 2*pi)
        angle = np.random.rand() * 2 * np.pi
        deltaX = np.cos(angle) * fiberLength
        deltaY = np.sin(angle) * fiberLength

        # Compute allowable start coordinates so that the entire fiber lies within the beam
        xMin = max(0, -deltaX)
        yMin = max(0, -deltaY)
        xMax = min(width, width - deltaX)
        yMax = min(height, height - deltaY)

        xStart = xMin + (xMax - xMin) * np.random.rand()
        yStart = yMin + (yMax - yMin) * np.random.rand()

        xEnd = xStart + deltaX
        yEnd = yStart + deltaY

        ax.plot([xStart, xEnd], [yStart, yEnd], color=fiberColor, linewidth=fiberWidth)
    
    # Draw top rebars
    if topCount > 1:
        topSpacing = (width - 2 * (cover + topDiameter / 2)) / (topCount - 1)
        for i in range(topCount):
            x = cover + topDiameter / 2 + i * topSpacing
            circle = patches.Circle((x, topRebarY), radius=topDiameter / 2,
                                    facecolor=rebarColor, edgecolor='k')
            ax.add_patch(circle)
    elif topCount == 1:
        x = width / 2
        circle = patches.Circle((x, topRebarY), radius=topDiameter / 2,
                                facecolor=rebarColor, edgecolor='k')
        ax.add_patch(circle)
    
    # Draw bottom rebars
    if botCount > 1:
        botSpacing = (width - 2 * (cover + botDiameter / 2)) / (botCount - 1)
        for i in range(botCount):
            x = cover + botDiameter / 2 + i * botSpacing
            circle = patches.Circle((x, botRebarY), radius=botDiameter / 2,
                                    facecolor=rebarColor, edgecolor='k')
            ax.add_patch(circle)
    elif botCount == 1:
        x = width / 2
        circle = patches.Circle((x, botRebarY), radius=botDiameter / 2,
                                facecolor=rebarColor, edgecolor='k')
        ax.add_patch(circle)
    
    ax.set_xlim([-cover, width + cover])
    ax.set_ylim([-cover, height + cover])
    
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('Beam Cross-Section')
    
    # --- Side View ---
    if sideview:
        ax2 = axs[1]
        ax2.clear()
        ax2.set_aspect('equal')
        concreteColor = (0.7, 0.7, 0.7)
        rebarColor = (0.8, 0.2, 0.2)
        fiberColor = (0, 0, 0)
        if loadingType is None:
            loadingType = 3
        if loadSpacing is None:
            loadSpacing = length / 2

        # Draw beam outline
        beam_rect = patches.Rectangle((0, 0), length, height, facecolor=concreteColor, edgecolor='k')
        ax2.add_patch(beam_rect)

        # Draw top rebars (as a band)
        if topCount > 0 and topDiameter > 0:
            topY = height - cover - topDiameter
            top_band = patches.Rectangle((0, topY), length, topDiameter, facecolor=rebarColor, edgecolor='none')
            ax2.add_patch(top_band)

        # Draw bottom rebars (as a band)
        if botCount > 0 and botDiameter > 0:
            botY = cover
            bot_band = patches.Rectangle((0, botY), length, botDiameter, facecolor=rebarColor, edgecolor='none')
            ax2.add_patch(bot_band)

        # Draw random fibers
        numFibers = int(mu * 200)
        fiberLength = height / 10
        fiberWidth = 0.5
        for _ in range(numFibers):
            xStart = np.random.rand() * length
            yStart = np.random.rand() * height
            angle = np.random.rand() * 2 * np.pi
            xEnd = xStart + np.cos(angle) * fiberLength
            yEnd = yStart + np.sin(angle) * fiberLength
            if 0 <= xEnd <= length and 0 <= yEnd <= height:
                ax2.plot([xStart, xEnd], [yStart, yEnd], color=fiberColor, linewidth=fiberWidth)

        # Draw loads (arrows)
        arrowLength = height / 4
        def drawArrow(x, y, dx, dy):
            ax2.arrow(x, y, dx, dy, head_width=height/15, head_length=height/10, fc='b', ec='b', linewidth=2, length_includes_head=True)

        if loadingType == 3:
            # Single point load at center
            drawArrow(length/2, height + arrowLength, 0, -arrowLength)
        else:
            # Two point loads
            firstLoadPosition = (length - loadSpacing) / 2
            secondLoadPosition = firstLoadPosition + loadSpacing
            drawArrow(firstLoadPosition, height + arrowLength, 0, -arrowLength)
            drawArrow(secondLoadPosition, height + arrowLength, 0, -arrowLength)

        # Draw support reaction arrows (always shown)
        supportArrowLength = height / 4
        drawArrow(0, -supportArrowLength, 0, supportArrowLength)  # left support
        drawArrow(length, -supportArrowLength, 0, supportArrowLength)  # right support
        drawArrow(length + supportArrowLength, 0, -supportArrowLength, 0)  # rightward roller

        # Set axis limits and labels
        ax2.set_xlim([-0.1*length, 1.2*length])
        ax2.set_ylim([-0.4*height, 1.5*height])
        ax2.set_xlabel('Length')
        ax2.set_ylabel('Height')
        ax2.set_title('2D Side View')
        #ax2.grid(True)

        plt.tight_layout()
        if show_fig:
            plt.show()
    else:
        if show_fig:
            plt.show()

