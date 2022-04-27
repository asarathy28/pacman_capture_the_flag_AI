
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
from distanceCalculator import Distancer

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'BusterAgent', second = 'BusterAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


'''**********************************************************************************************'''

def getNonWalls(gameState):
    spots = []
    walls = gameState.getWalls().asList()

    rightMost = max([wall[0] for wall in walls])
    upBound = max([wall[1] for wall in walls])
    lowBound = min([wall[1] for wall in walls])

    for x in range(0, rightMost):
      for y in range(lowBound, upBound):
        spots.append((x, y))

    nonWalls = []
    for spot in spots:
        if spot not in gameState.getWalls().asList():
            nonWalls.append(spot)
    return nonWalls

def findDeadEndPoints(walls, nonWalls):
    deadEnds = []
    for pos in nonWalls:
        if surroundingWalls(pos, walls) == 3:
            deadEnds.append(pos)

    return deadEnds

def surroundingWalls(position, walls):
    num = 0
    if (position[0]+1, position[1]) in walls:
        num+=1
    if (position[0]-1, position[1]) in walls:
        num+=1
    if (position[0], position[1]+1) in walls:
        num+=1
    if (position[0], position[1]-1) in walls:
        num+=1
    return num

def boarderCoordinateX (width, isRed):
    halfway = width/2
    if isRed: return halfway
    else: return halfway+1

class InferAgent (CaptureAgent):
    "An agent that tracks and displays its beliefs about ghost positions."

    def registerInitialState(self, gameState, inference = "ParticleFilter", observeEnable = True, elapseTimeEnable = True):

        CaptureAgent.registerInitialState(self, gameState)

        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(self.index, a, gameState) for a in self.getOpponents(gameState)]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        "Initializes beliefs and inference modules"
        import __main__
        #self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.invaderBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True
        self.deadEndPoints = findDeadEndPoints(gameState.getWalls().asList(), getNonWalls(gameState))
        walls = gameState.getWalls().asList()
        width = max([wall[0] for wall in walls])
        self.halfway = 0
        if self.red: self.halfway = width/2.0
        else: self.halfway = width/2.0 + 1.0





    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."

        for index, inf in enumerate(self.inferenceModules):
            if not self.firstMove and self.elapseTimeEnable:
                inf.elapseTime(gameState)
            self.firstMove = False
            if self.observeEnable:
                inf.observeState(gameState)
            self.invaderBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.invaderBeliefs)
        self.displayDistributionsOverPositions(self.invaderBeliefs)


        return CaptureAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        pass


    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor





class BusterAgent (InferAgent):

    def chooseAction(self, gameState):

        hunterPosition = gameState.getAgentPosition(self.index)
        opponentsPositionDistributions = [beliefs for i, beliefs in enumerate(self.invaderBeliefs)]

        closestOpptPos = None
        mazeDisToClosestOpp = float('inf')
        OppPos = None
        mazeDisToOpp = None
        #loop though distribution for each ghost and find closest ghost
        for opd in opponentsPositionDistributions:

            highestProb = 0
            for pos, prob in opd.items():
                if prob>highestProb:
                    highestProb=prob
                    mazeDisToOpp=self.distancer.getDistance(pos, hunterPosition)
                    OppPos=pos
            if mazeDisToOpp<mazeDisToClosestOpp:
                mazeDisToClosestOpp=mazeDisToOpp
                closestOppPos=OppPos


        legal = [a for a in gameState.getLegalActions(self.index)]
        bestAction=None
        bestEval=-99999999
        #print gameState.data.layout
        for action in legal:
            successor = self.getSuccessor(gameState, action)
            successorPosition = successor.getAgentPosition(self.index)
            tempEval = self.evaluationFunction(successor)


            if tempEval>bestEval:
                bestEval=tempEval
                bestAction=action

        '''
        print "Best Action: ", bestAction
        print "Best Eval: ", bestEval
        print "Maze Dist: ", mazeDisToClosestOpp
        '''
        if closestOppPos in self.deadEndPoints:
            if mazeDisToClosestOpp < 2:
                print "CORNERED!!"

                bestAction = 'Stop'


        return bestAction


    def evaluationFunction(self,gameState):

        eval = 0


        hunterPosition = gameState.getAgentPosition(self.index)
        legal = [a for a in gameState.getLegalActions(self.index)]

        opponentsPositionDistributions = [beliefs for i, beliefs in enumerate(self.invaderBeliefs)]

        closestOpptPos = None
        mazeDisToClosestOpp = float('inf')
        OppPos = None
        mazeDisToOpp = None
        #loop though distribution for each ghost and find closest ghost
        for opd in opponentsPositionDistributions:

            highestProb = 0
            for pos, prob in opd.items():

                if prob>highestProb:
                    highestProb=prob
                    mazeDisToOpp=self.distancer.getDistance(pos, hunterPosition)
                    OppPos=pos
            if mazeDisToOpp<mazeDisToClosestOpp:
                mazeDisToClosestOpp=mazeDisToOpp
                closestOppPos=OppPos


        eval = 1000.0/(1+mazeDisToClosestOpp)
        scaredTimes = gameState.getAgentState(self.index).scaredTimer

        if scaredTimes > 0:
            if mazeDisToClosestOpp < 2:
                eval -= 1000000

        # punish if we go on the other side
        diff = hunterPosition[0] - self.halfway
        if self.red:
            if diff > 0:
                punish = diff*(-100.0)
                eval += punish
        else:
            if diff < 0:
                punish = diff*(100.0)
                eval += punish

        
        if hunterPosition == gameState.getInitialAgentPosition(self.index):
            return -1000000

        # reward if close to ghost
        return eval












''' INFERENCE '''
'''**********************************************************************************************'''


def getSucc(pos, gameState):
    me = pos
    validSucc = []
    if (me[0]+1,me[1]) not in gameState.getWalls().asList():
        validSucc.append((me[0]+1,me[1]))
    if (me[0]-1,me[1]) not in gameState.getWalls().asList():
        validSucc.append((me[0]-1,me[1]))
    if (me[0],me[1]+1) not in gameState.getWalls().asList():
        validSucc.append((me[0],me[1]+1))
    if (me[0],me[1]-1) not in gameState.getWalls().asList():
        validSucc.append((me[0],me[1]-1))
    # append current position to represent direction.stop
    validSucc.append((me[0],me[1]))

    return validSucc


class InferenceModule:

    def __init__(self, aIndex, oppIndex, gameState):
        "Sets the ghost agent for later access"
        self.oppAgent = gameState.getAgentState(oppIndex)
        self.agentIndex = aIndex
        self.index = oppIndex
        self.obs = [] # most recent observation position



    def getPositionDistribution(self, gameState, oppPosition):
        ''' do this by hand '''

        actionDist = util.Counter()
        legalActions = getSucc(oppPosition, gameState)
        for a in legalActions: actionDist[a] = 1.0
        actionDist.normalize()

        '''
        dist = util.Counter()

        for action, prob in actionDist.items():
            successor = gameState.generateSuccessor(self.oppIndex, action)
            successorPosition = successor.getAgentState(self.oppIndex).getPosition()
            dist[successorPosition] = prob
        '''

        return actionDist

    def observeState(self, gameState):
        "Collects the relevant noisy distance observation and pass it along."
        distances = gameState.getAgentDistances()
        if len(distances) > self.index: # Check for missing observations
            obs = distances[self.index]
            self.obs = obs
            self.observe(obs, gameState)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.initializeUniformly(gameState)


    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        "Sets the belief state to a uniform prior belief over all positions."
        pass

    def observe(self, observation, gameState):
        "Updates beliefs based on the given distance observation and gameState."
        pass

    def elapseTime(self, gameState):
        "Updates beliefs for a time step elapsing from a gameState."
        pass

    def getBeliefDistribution(self):
        """
        Returns the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        pass



class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.

    Useful helper functions will include random.choice, which chooses an element
    from a list uniformly at random, and util.sample, which samples a key from a
    Counter by treating its values as probabilities.
    """

    def __init__(self, aIndex, oppAgent, gs, numParticles=300):
        InferenceModule.__init__(self, aIndex, oppAgent, gs);
        self.setNumParticles(numParticles)
        self.canSeeInvader = False
        self.invaderRealPos = None



    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        self.particles = []
        for i in range(self.numParticles):
            self.particles.append([self.legalPositions[i % len(self.legalPositions)], 1])

    def observe(self, observation, gameState):

        noisyDistance = observation
        agentPosition = gameState.getAgentPosition(self.agentIndex)


        if self.canSeeInvader:
            for i in range(self.numParticles):
                self.particles[i] = [self.invaderRealPos, 1]
        else:
            allZero = True
            for i in range(self.numParticles):
                self.particles[i][1] = gameState.getDistanceProb(util.manhattanDistance(self.particles[i][0], agentPosition), noisyDistance)
                if self.particles[i][1] != 0:
                    allZero = False

            if allZero:
                self.initializeUniformly(gameState)

            else:
                beliefDistrib = self.getBeliefDistribution()
                for i in range(self.numParticles):
                    self.particles[i] = [util.sample(beliefDistrib), 1]


    def elapseTime(self, gameState):
        if gameState.getAgentPosition(self.index):
            self.canSeeInvader = True
            self.invaderRealPos = gameState.getAgentPosition(self.index)

        else:
            self.canSeeInvader = False

        for i in range(self.numParticles):
            newPosDist = self.getPositionDistribution(gameState, self.particles[i][0])
            self.particles[i][0]=util.sample(newPosDist)


    def getBeliefDistribution(self):
        beliefs = util.Counter()
        for i in range(self.numParticles):
            beliefs[self.particles[i][0]] += self.particles[i][1]
        beliefs.normalize()
        return beliefs
        #util.raiseNotDefined()
