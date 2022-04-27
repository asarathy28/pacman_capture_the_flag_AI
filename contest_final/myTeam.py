# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgentTwo', second = 'DefensiveReflexAgent'):
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
  returnList = []
  returnList.append(eval(first)(firstIndex))
  returnList.append(eval(second)(secondIndex))
  return returnList

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    self.start = gameState.getAgentPosition(self.index)
    self.foodToStart = len(self.getFood(gameState).asList())
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''

  def chooseAction(self, gameState):
    """
    Picks actions with the highest Q(s, a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    onOppSide = False
    if self.index % 2 == 1:
      if not gameState.isRed(gameState.getAgentPosition(self.index)):
        onOppSide = True
    if self.index % 2 == 0:
      if gameState.isRed(gameState.getAgentPosition(self.index)):
        onOppSide = True


    if foodLeft <= 2 and not onOppSide:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

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

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(DummyAgent):

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    opponentIndices = self.getOpponents(gameState)
    ourIndices = self.getTeam(gameState)

    noisyDistances = successor.getAgentDistances()
    opponentNoisyDistances = []
    ourDistances = []

    for index in opponentIndices:
      opponentNoisyDistances.append(noisyDistances[index])
    for index in ourIndices:
      ourDistances.append(noisyDistances[index])

    smallestOppDist = 9999
    for ourDist in ourDistances:
      for theirDist in opponentNoisyDistances:
        if abs(ourDist-theirDist) < smallestOppDist:
          smallestOppDist = abs(ourDist-theirDist)
    features['distanceToOpponent'] = smallestOppDist

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    features['ghostsNearby'] = len(ghosts)

    if len(ghosts) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts])
      features['distanceToGhost'] = minDistance
    else:
      features['distanceToGhost'] = 999

    attackingCapsules = self.getCapsules(gameState)
    if len(attackingCapsules) > 0:
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, cap) for cap in attackingCapsules])
      #features['powerPellet'] = minDistance


    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'distanceToGhost': 10000,'ghostsNearby':-1000000}

class DefensiveReflexAgent(DummyAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
      features = util.Counter()
      successor = self.getSuccessor(gameState, action)

      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()

      # Computes whether we're on defense (1) or offense (0)
      features['onDefense'] = 1
      if myState.isPacman: features['onDefense'] = 0

      # Computes distance to invaders we can see
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      features['numInvaders'] = len(invaders)
      if len(invaders) > 0:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        features['invaderDistance'] = min(dists)

      if action == Directions.STOP: features['stop'] = 1
      rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      if action == rev: features['reverse'] = 1

      return features

    def getWeights(self, gameState, action):
      return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


class OffensiveReflexAgentTwo(CaptureAgent):

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    self.start = gameState.getAgentPosition(self.index)
    self.foodToStart = len(self.getFood(gameState).asList())
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    self.foodListToStart = self.getFood(gameState).asList()
    self.startOverFood = self.foodToStart
    self.actionsToHome = util.Queue()
    self.actionsToFood = util.Queue()
    self.foodPaths = {}
    self.debugCells = []
    self.stateChoices = {}
    self.stateHomeChoices = {}
    print gameState.getWalls()

    self.homeBorderX = 0
    halfway = gameState.data.layout.width/2
    if self.red:
      self.homeBorderX = halfway-1
    else:
      self.homeBorderX = halfway

    self.opposingTeamLocations = self.getTheirSpots(gameState.getWalls().asList())
    self.opposingTeamNonWalls = self.getNonWalls(self.opposingTeamLocations, gameState)
    self.deadEnds = self.findDeadEnds(gameState.getWalls().asList(), self.opposingTeamNonWalls)
    print "deadEnds", self.deadEnds
    self.walls = gameState.getWalls().asList()


    #for food in self.foodListToStart:
      #self.foodPaths[food] = self.initialAStar(gameState, food)

    #for key, val in self.foodPaths:
      #print key, val

  def getTheirSpots(self, walls):
    rightMost = max([wall[0] for wall in walls])
    upBound = max([wall[1] for wall in walls])
    lowBound = min([wall[1] for wall in walls])
    theirSpots = []

    if self.red:
      rightBound = max([wall[0] for wall in walls])
      leftBound = rightMost/2+1
    else:
      rightBound = rightMost/2
      leftBound = 0

    for x in range(leftBound, rightBound):
      for y in range(lowBound, upBound):
        theirSpots.append((x, y))
    return theirSpots

  def getNonWalls(self, spots, gameState):
    nonWalls = []
    for spot in spots:
      if spot not in gameState.getWalls().asList():
        nonWalls.append(spot)
    return nonWalls

  def findDeadEnds(self, walls, nonWalls):
    deadEndRoads = []
    deadEndPoints = []
    for pos in nonWalls:
      if self.surroundingWalls(pos, walls) == 3:
        deadEndPoints.append(pos)

    print "deadEndPoints", deadEndPoints
    for point in deadEndPoints:
      alreadyAccountedFor = False
      for val in deadEndRoads:
        if point in val:
          alreadyAccountedFor = True
      if not alreadyAccountedFor:
        tunnelList = [(point)]
        chosenPoint = self.determineChosenPoints(point, walls, tunnelList)
        pointsToEvaluate = util.Queue()
        while self.surroundingWalls(chosenPoint, walls) == 2:
          print "chosenPoint", chosenPoint
          tunnelList.append(chosenPoint)
          chosenPoint = self.determineChosenPoints(chosenPoint, walls, tunnelList)
        deadEndRoads.append(tunnelList)
    return deadEndRoads

  def determineChosenPoints(self, point, walls, tunnelList):
    availablePoints = []
    if (point[0]+1, point[1]) not in walls and (point[0]+1, point[1]) not in tunnelList:
      return (point[0]+1, point[1])
    if (point[0]-1, point[1]) not in walls and (point[0]-1, point[1]) not in tunnelList:
      return (point[0]-1, point[1])
    if (point[0], point[1]+1) not in walls and (point[0], point[1]+1) not in tunnelList:
      return (point[0], point[1]+1)
    if (point[0], point[1]-1) not in walls and (point[0], point[1]-1) not in tunnelList:
      return (point[0], point[1]-1)
    return availablePoints

  def isTunnelPoint(self, point, walls, tunnelList):
    if self.surroundingWalls(point, walls) >= 2:
      return True
    if self.surroundingWalls((point[0]+1, point[1]), walls) < 2 \
      and (point[0]+1, point[1]) not in tunnelList:
      return False
    if self.surroundingWalls((point[0] - 1, point[1]), walls) < 2 \
            and (point[0] - 1, point[1]) not in tunnelList:
      return False
    if self.surroundingWalls((point[0], point[1]+1), walls) < 2 \
            and (point[0], point[1]+1) not in tunnelList:
      return False
    if self.surroundingWalls((point[0], point[1]-1), walls) < 2 \
            and (point[0], point[1]-1) not in tunnelList:
      return False

    return True

  def surroundingWalls(self, position, walls):
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


  def initialAStar(self, gameState, food):
    ourNode = node(None, None, gameState)
    frontier = util.PriorityQueue()
    explored = []
    a = True

    startNodePos = ourNode.getState().getAgentPosition(self.index)
    frontier.push(ourNode, self.getMazeDistance(startNodePos, food))

    while(a):
      if frontier.isEmpty():
        return []
      ourNode = frontier.pop()
      if gameState.getAgentPosition(self.index) == food:
        returnList = []
        while ourNode.getParent():
          parent = ourNode.getParent()
          returnList.append(parent.getDirection())
        returnList.reverse()
        returnList.remove(None)
        return returnList

      nodeState = ourNode.getState()
      explored.append(ourNode.getState())
      for action in nodeState.getLegalActions(self.index):
        successor = nodeState.generateSuccessor(self.index, action)
        childNode = node(ourNode, action, successor)
        if childNode.getState() not in explored:
          childNodePos = childNode.getState().getAgentPosition(self.index)
          frontier.push(childNode, self.getMazeDistance(childNodePos, food))

  def updateDict(self, gameState, action):
    self.stateChoices[gameState] = action

  def updateHomeDict(self, gameState, action):
    self.stateHomeChoices[gameState] = action

  def getOptimalAction(self, gameState):
    return self.stateChoices[gameState]

  def getOptimalHomeAction(self, gameState):
    return self.stateHomeChoices[gameState]

  def setStartOverFood(self, food):
    self.startOverFood = food

  def setActionsToHome(self, directions):
    for item in directions:
      self.actionsToHome.push(item)

  def setActionsToFood(self, directions):
    for item in directions:
      self.actionsToFood.push(item)

  def aStarDeadEnd(self, initial_pos, entrance):
    startNode = node3(initial_pos, None)
    frontier = util.PriorityQueue()
    print "startNode", startNode.getPosition(), "entrance", entrance
    frontier.push(startNode, self.getMazeDistance(startNode.getPosition(), self.start))
    explored = []
    a = True
    while(a):
      if frontier.isEmpty():
        print "failure"
        returnList = []
        while lastPopped.getParent():
          returnList.append(lastPopped.getPosition())
          lastPopped = lastPopped.getParent()
        returnList.reverse()
        return returnList
      ourNode = frontier.pop()
      lastPopped = ourNode
      if ourNode.getPosition() not in self.opposingTeamLocations:
        returnList = []
        while ourNode.getParent():
          returnList.append(ourNode.getPosition())
          ourNode = ourNode.getParent()
        returnList.reverse()
        return returnList

      explored.append(ourNode.getPosition())
      successors = self.getNeighboringPositions(ourNode.getPosition(), entrance, self.walls)
      for successor in successors:
        childNode = node3(successor, ourNode)
        if childNode.getPosition() not in explored:
          frontier.push(childNode, self.getMazeDistance(childNode.getPosition(), self.start))

  def getNeighboringPositions(self, position, entrance, walls):
    neighboringPositions = []
    if (position[0]+1, position[1]) not in walls and (position[0]+1, position[1]) != entrance:
      neighboringPositions.append((position[0]+1, position[1]))
    if (position[0]-1, position[1]) not in walls and (position[0]-1, position[1]) != entrance:
      neighboringPositions.append((position[0]-1, position[1]))
    if (position[0], position[1]+1) not in walls and (position[0], position[1]+1) != entrance:
      neighboringPositions.append((position[0], position[1]+1))
    if (position[0], position[1]-1) not in walls and (position[0], position[1]-1) != entrance:
      neighboringPositions.append((position[0], position[1]-1))
    return neighboringPositions
  def dangerDeadEnd(self, successor, current):
    enteredTunnel = False
    currentlyInTunnel = False
    tunnels = []

    for road in self.deadEnds:
      if successor.getAgentPosition(self.index) in road:
        enteredTunnel = True
        tunnels.append(road)

    for road in self.deadEnds:
      if current.getAgentPosition(self.index) in road:
        currentlyInTunnel = True
        tunnels.append(road)

    if not enteredTunnel:
      return False

    largestTunnel = tunnels[0]
    for tunnel in tunnels:
      if len(tunnel) > len(largestTunnel):
        largestTunnel = tunnel

    foodInLargestTunnel = []
    for pos in largestTunnel:
      if pos in self.getFood(current).asList():
        print pos, "is food"
        foodInLargestTunnel.append(pos)

    if len(foodInLargestTunnel) == 0:
      if not currentlyInTunnel:
        return True
      else:
        return False

    farthestFood = foodInLargestTunnel[0]
    farthestDistance = self.getMazeDistance(current.getAgentPosition(self.index), foodInLargestTunnel[0])
    for food in foodInLargestTunnel:
      if self.getMazeDistance(current.getAgentPosition(self.index), food) > farthestDistance:
        farthestDistance = self.getMazeDistance(current.getAgentPosition(self.index), food)

    opponents = self.getOpponents(current)
    opponentDistances = [current.getAgentDistances()[o] for o in opponents]
    print "opponentDistances", opponentDistances, "farthestDistance", farthestDistance
    for dist in opponentDistances:
      if dist <= farthestDistance*2:
        return True

    return False


  def chooseAction(self, gameState):
    ourPos = gameState.getAgentPosition(self.index)
    if self.home(gameState):
      self.setStartOverFood(len(self.getFood(gameState).asList()))



    if len(self.actionsToHome.list) > 0:
      popped = self.actionsToHome.pop()
      return popped

    if len(self.actionsToFood.list) > 0:
      popped = self.actionsToFood.pop()
      return popped

    # if we've collected 5 food, aStarHome
    if len(self.getFood(gameState).asList()) <= self.startOverFood-5 and not self.home(gameState):
      if len(self.getObservableGhosts(gameState)) > 0:
        if gameState.getAgentState(self.getObservableGhosts(gameState)[0]).scaredTimer==0:
          gameAgents = []
          gameAgents.append(self.index)
          for g in self.getObservableGhosts(gameState):
            gameAgents.append(g)
          maxVal = self.maxHomeValue(gameState, float('-inf'), float('inf'), gameAgents, 2, 0)
          return self.getOptimalHomeAction(gameState)

      legalActions = gameState.getLegalActions(self.index)
      bestOne = legalActions[0]
      evals = []
      evalDict = {}
      largestEval = self.evaluationHomeFunction(gameState.generateSuccessor(self.index, bestOne))
      for action in legalActions:
        successor = gameState.generateSuccessor(self.index, action)
        tempEval = self.evaluationHomeFunction(successor)
        if tempEval > largestEval:
          largestEval = tempEval
          bestOne = action
      return bestOne

    else:
      if len(self.getObservableGhosts(gameState)) > 0:
        if gameState.getAgentState(self.getObservableGhosts(gameState)[0]).scaredTimer==0:
          gameAgents = []
          gameAgents.append(self.index)
          for g in self.getObservableGhosts(gameState):
            gameAgents.append(g)
            maxVal = self.maxValue(gameState, float('-inf'), float('inf'), gameAgents, 2, 0)
            return self.getOptimalAction(gameState)

      #IF WE DON'T GO HOME, CONTINUE COLLECTING FOOD IN ENEMY TERRITORY
      legalActions = gameState.getLegalActions(self.index)

      for action in legalActions:
        initialPos = gameState.getAgentPosition(self.index)
        entrance = gameState.generateSuccessor(self.index, action).getAgentPosition(self.index)
        print "pathHome", self.aStarDeadEnd(initialPos, entrance)
        if self.dangerDeadEnd(gameState.generateSuccessor(self.index, action), gameState):
          legalActions.remove(action)


      bestOne = legalActions[0]
      evals = []
      evalDict = {}
      largestEval = self.evaluationFunction(gameState.generateSuccessor(self.index, bestOne))
      for action in legalActions:
        successor = gameState.generateSuccessor(self.index, action)
        tempEval = self.evaluationFunction(successor)
        if tempEval > largestEval:
          largestEval = tempEval
          bestOne = action
      return bestOne

  def home(self, gameState):
    if self.red and gameState.getAgentPosition(self.index)[0] <= self.homeBorderX \
      or not self.red and gameState.getAgentPosition(self.index)[0] >= self.homeBorderX:
        return True
    return False

  def maxValue(self, gameState, alpha, beta, agents, depth, index):
    if len(self.getObservableGhosts(gameState)) < len(agents)-1:
      return self.evaluationFunction(gameState)

    if len(self.getFood(gameState).asList()) <= self.startOverFood-5 or self.lost(gameState) or index==depth*len(agents):
      return self.evaluationFunction(gameState)

    legalActions = gameState.getLegalActions(self.index)

    for action in legalActions:
      initialPos = gameState.getAgentPosition(self.index)
      entrance = gameState.generateSuccessor(self.index, action).getAgentPosition(self.index)
      print "pathHome", self.aStarDeadEnd(initialPos, entrance)
      if self.dangerDeadEnd(gameState.generateSuccessor(self.index, action), gameState):
        print "danger if we move", action
        legalActions.remove(action)

    v = float('-inf')
    bestAction = legalActions[0]

    for action in legalActions:
      prev_v = v
      v = max(v, self.minValue(gameState.generateSuccessor(self.index, action), alpha, beta, agents, depth, index+1))
      if prev_v != v:
        bestAction = action
      if v > beta:
        self.updateDict(gameState, bestAction)
        return v
      alpha = max(alpha, v)
    self.updateDict(gameState, bestAction)
    return v
    util.raiseNotDefined()

  def minValue(self, gameState, alpha, beta, agents, depth, index):
    if len(self.getObservableGhosts(gameState)) < len(agents) - 1:
      return self.evaluationFunction(gameState)

    if len(self.getFood(gameState).asList()) <= self.startOverFood - 5 or self.lost(gameState) or index == depth * len(
            agents):
      return self.evaluationFunction(gameState)

    div1 = index / len(agents)
    mod1 = index - div1 * len(agents)
    player = agents[mod1]

    div2 = (index+1) / len(agents)
    mod2 = (index+1) - div2 * len(agents)
    nextPlayer = agents[mod2]

    legalActions = gameState.getLegalActions(player)

    v = float('inf')
    bestAction = legalActions[0]

    for action in legalActions:
      if nextPlayer == agents[0]:
        prev_v = v
        v = min(v, self.maxValue(gameState.generateSuccessor(player, action), alpha, beta, agents, depth, index+1))
        if prev_v != v:
          bestAction = action
        if v < alpha:
          self.updateDict(gameState, bestAction)
          return v
        beta = min(beta, v)
      else:
        prev_v = v
        v = min(v, self.minValue(gameState.generateSuccessor(player, action), alpha, beta, agents, depth, index + 1))
        if prev_v != v:
          bestAction = action
        if v < alpha:
          self.updateDict(gameState, bestAction)
          return v
        beta = min(beta, v)
    self.updateDict(gameState, bestAction)
    return v
    util.raiseNotDefined()

  def maxHomeValue(self, gameState, alpha, beta, agents, depth, index):

      if len(self.getObservableGhosts(gameState)) < len(agents)-1:
        return self.evaluationHomeFunction(gameState)

      if self.home(gameState) or self.lost(gameState) or index==depth*len(agents):
        return self.evaluationHomeFunction(gameState)

      legalActions = gameState.getLegalActions(self.index)

      v = float('-inf')
      bestAction = legalActions[0]

      for action in legalActions:
        prev_v = v
        v = max(v, self.minHomeValue(gameState.generateSuccessor(self.index, action), alpha, beta, agents, depth, index+1))
        if prev_v != v:
          bestAction = action
        if v > beta:
          self.updateHomeDict(gameState, bestAction)
          return v
        alpha = max(alpha, v)
      self.updateHomeDict(gameState, bestAction)
      return v
      util.raiseNotDefined()

  def minHomeValue(self, gameState, alpha, beta, agents, depth, index):
    if len(self.getObservableGhosts(gameState)) < len(agents) - 1:
      return self.evaluationHomeFunction(gameState)

    if self.home(gameState) or self.lost(gameState) or index == depth * len(
            agents):
      return self.evaluationHomeFunction(gameState)

    div1 = index / len(agents)
    mod1 = index - div1 * len(agents)
    player = agents[mod1]

    div2 = (index+1) / len(agents)
    mod2 = (index+1) - div2 * len(agents)
    nextPlayer = agents[mod2]

    legalActions = gameState.getLegalActions(player)

    v = float('inf')
    bestAction = legalActions[0]

    for action in legalActions:
      if nextPlayer == agents[0]:
        prev_v = v
        v = min(v, self.maxHomeValue(gameState.generateSuccessor(player, action), alpha, beta, agents, depth, index+1))
        if prev_v != v:
          bestAction = action
        if v < alpha:
          self.updateHomeDict(gameState, bestAction)
          return v
        beta = min(beta, v)
      else:
        prev_v = v
        v = min(v, self.minHomeValue(gameState.generateSuccessor(player, action), alpha, beta, agents, depth, index + 1))
        if prev_v != v:
          bestAction = action
        if v < alpha:
          self.updateHomeDict(gameState, bestAction)
          return v
        beta = min(beta, v)
    self.updateHomeDict(gameState, bestAction)
    return v
    util.raiseNotDefined()



  # FUNCTION THAT RETURNS WHETHER OR NOT I SHATTERED
  def lost(self, gameState):
    if gameState.getAgentState(self.index).configuration.getPosition() == self.start:
      return True
    return False


  #FUNCTION THAT RETURNS A LIST OF OBSERVABLE GHOSTS WHEN I'M ON OFFENSE (WHEN I'M PACMAN)
  def getObservableGhosts(self, gameState):
    opponents = self.getOpponents(gameState)
    observableGhosts = [o for o in opponents if not gameState.getAgentState(o).isPacman
                        and gameState.getAgentPosition(o)]
    return observableGhosts



  def aStarHome(self, gameState):
    ourNode = node(None, None, gameState)
    frontier = util.PriorityQueue()
    explored = []
    a = True

    startNodePos = ourNode.getState().getAgentPosition(self.index)
    frontier.push(ourNode, self.getMazeDistance(startNodePos, self.start))

    while (a):
      if frontier.isEmpty():
        a = False
        return []
      ourNode = frontier.pop()
      #print "popped", ourNode.getState().getAgentPosition(self.index)
      nodeState = ourNode.getState()
      if nodeState.getAgentPosition(self.index)[0] == self.homeBorderX:
        returnList = []
        while ourNode.getParent():
          returnList.append(ourNode.getDirection())
          ourNode = ourNode.getParent()
        returnList.reverse()
        return returnList

      nodeState = ourNode.getState()
      explored.append(ourNode.getState())
      legalActions = nodeState.getLegalActions(self.index)
      opponents = self.getOpponents(nodeState)
      opponentPositions = []
      for opponent in opponents:
        opponentPositions.append(nodeState.getAgentPosition(opponent))

      for action in legalActions:
        if nodeState.generateSuccessor(self.index, action).getAgentState\
                  (self.index).configuration.getPosition() in opponentPositions:
          if len(legalActions) > 1:
            legalActions.remove(action)

      for action in legalActions:
        successor = nodeState.generateSuccessor(self.index, action)
        childNode = node(ourNode, action, successor)
        if childNode.getState() not in explored:
          childNodePos = childNode.getState().getAgentPosition(self.index)
          frontier.push(childNode, self.getMazeDistance(childNodePos, self.start))

  def aStarFood(self, gameState):
    ourNode = node2(None, None, gameState, gameState.getAgentPosition(self.index), len(self.getFood(gameState).asList()))
    frontier = util.PriorityQueue()
    explored = []
    a = True

    startNodePos = ourNode.getPosition()
    frontier.push(ourNode, self.farthestFood(startNodePos, ourNode.getState()))

    while (a):
      if frontier.isEmpty():
        a = False
        return []
      ourNode = frontier.pop()
      nodeState = ourNode.getState()
      if ourNode.getFoodLeft() == 2:
        returnList = []
        while ourNode.getParent():
          returnList.append(ourNode.getDirection())
          ourNode = ourNode.getParent()
        returnList.reverse()
        return returnList

      nodeState = ourNode.getState()
      explored.append((ourNode.getPosition(), ourNode.getFoodLeft()))

      for action in nodeState.getLegalActions(self.index):
        successor = nodeState.generateSuccessor(self.index, action)
        position = successor.getAgentPosition(self.index)

        foodLeft = len(self.getFood(successor).asList())
        childNode = node2(ourNode, action, successor, position, foodLeft)
        if (childNode.getPosition(), childNode.getFoodLeft()) not in explored:
          childNodePos = childNode.getPosition()
          frontier.push(childNode, self.farthestFood(childNodePos, childNode.getState())[1]
                        + self.getMazeDistance(childNodePos, childNode.getParent().getPosition()))

  def farthestFood(self, position, gameState):
    tempQueue = util.PriorityQueue()
    for food in self.getFood(gameState).asList():
      tempQueue.push((food, -1*self.getMazeDistance(position, food)), -1*self.getMazeDistance(position, food))
    tempQueue.pop()
    if tempQueue.count == 1:
      return tempQueue.pop()
    return tempQueue.pop()

  def evaluationFunction(self, gameState):
    currPos = gameState.getAgentPosition(self.index)
    currFood = self.getFood(gameState)
    currFoodList = currFood.asList()
    opponents = self.getOpponents(gameState)
    currScaredTimes = []

    for opp in opponents:
      currScaredTimes.append(gameState.getAgentState(opp).scaredTimer)

    if len(currFoodList) > 0:  # This should always be True,  but better safe than sorry
      myPos = gameState.getAgentPosition(self.index)
      minDistance = min([self.getMazeDistance(myPos, food) for food in currFoodList])
    else:
      minDistance = 0.9

    shortestGhost = 9999
    if len(self.getObservableGhosts(gameState)) > 0:
      shortestGhost = min([self.getMazeDistance(currPos, gameState.getAgentPosition(o)) for o
                           in self.getObservableGhosts(gameState)])

    totalScared = 0
    if len(currScaredTimes) > 0:
      for item in currScaredTimes:
        totalScared+=item

    if currPos == self.start or shortestGhost==0:
      return -100000
    if minDistance==0 or len(currFoodList) == 0:
      return 100000
    if minDistance > 0:
      return 10000*(1.0/float(len(currFoodList))) + 10*(1.0/float(minDistance)) + totalScared

  def evaluationHomeFunction(self, gameState):
    currPos = gameState.getAgentPosition(self.index)
    if currPos == self.start:
      return -100000
    if self.home(gameState):
      return 100000
    return 1.0/self.getMazeDistance(currPos, self.start)

  def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


  def aStarSearch(self, gameState):
    """Search the node that has the lowest combined cost and heuristic first."""
    # creates a start node
    startNode = node(gameState, None, 0, None, self.heuristic(gameState, self.index))
    # create dictionary to store states and path costs of nodes in the priority queue
    inPQ = {}
    # creates a priority queue for the frontier
    astarPQ = util.PriorityQueue()
    # push start node onto priority queue
    astarPQ.push(startNode, (startNode.getPathCost() + startNode.getHeuristic()))
    # track what's in the priority queue by adding the path cost of the start node
    inPQ[startNode.stateToString()] = startNode.getPathCost() + startNode.getHeuristic()
    # create empty list to store visited states
    exploredSet = []
    a = True
    # create empty list to store return list of actions
    returnList = []
    # while goal state has not been found and there are still nodes to explore
    while (a):
      # if the priority queue is empty, then pacman has not found a solution
      if astarPQ.isEmpty():
        a = False
        return []
      # pop top priority node from priority queue, i.e. with lowest f(n)
      currentNode = astarPQ.pop()
      print "choosing heuristic", currentNode.getHeuristic()
      # if the current node is the goal state
      print "red team?:", self.red, "red side of the board right now?:", currentNode.getState().isRed(gameState.getAgentPosition(self.index))
      if (self.red and currentNode.getState().isRed(currentNode.getState().getAgentPosition(self.index)) or \
        (not self.red and not currentNode.getState().isRed(currentNode.getState().getAgentPosition(self.index)))) and \
              currentNode.getState().getAgentPosition(self.index) != self.start:
        print "made it to the goal"
        # add direction of current node to return list of actions
        returnList.append(currentNode.getDirection())
        # while current node has a parent node
        while currentNode.getParent():
          # get parent node of current node and call it temp node
          tempNode = currentNode.getParent()
          # add direction of parent node to return list of actions
          returnList.append(tempNode.getDirection())
          # rename current node reference to parent node from current node
          currentNode = currentNode.getParent()
        # remove direction of start node from return list of actions
        returnList.remove(None)
        # reverse return list of actions
        returnList.reverse()
        # return list of actions
        return returnList
      # add state of current node to explored set of node states
      exploredSet.append(currentNode.getState())
      # for each child node of current node
      successors = {}
      for action in gameState.getLegalActions(self.index):
        successors[gameState.generateSuccessor(self.index, action)] = action
      for i in successors.keys():
        # create child node using node class
        childNode = node(i, successors[i], 0, currentNode, self.heuristic(i, self.index))
        # if state of child node is not in the explored set of states and dictionary
        if childNode.getState() not in exploredSet and childNode.stateToString() not in inPQ.keys():
          # push child node with path cost + heuristic cost onto priority queue
          astarPQ.push(childNode, (childNode.getPathCost() + childNode.getHeuristic()))
          # add child node with path cost + heuristic cost to dictionary
          inPQ[childNode.stateToString()] = childNode.getPathCost() + childNode.getHeuristic()
        # if state of child node is in the dictionary and path cost + heuristic cost is less than current function
        elif childNode.stateToString() in inPQ.keys() and (childNode.getPathCost() + childNode.getHeuristic()) < \
                inPQ[childNode.stateToString()]:
          # update child node in priority queue with updated path cost + heuristic cost
          astarPQ.update(childNode, (childNode.getPathCost() + childNode.getHeuristic()))
    util.raiseNotDefined()


  # Abbreviations
  astar = aStarSearch

  def heuristic(self, gameState, index):
    myPos = gameState.getAgentPosition(index)

    if myPos == self.start:
      return 10000

    if self.red and gameState.isRed(gameState.getAgentPosition(self.index)) or \
            (not self.red and not gameState.isRed(gameState.getAgentPosition(self.index))):
      return 0
    return abs(myPos[0]-self.start[0])*-1

class node():
  def __init__(self, parent, direction, state):
    self.parent = parent
    self.direction = direction
    self.state = state

  def getState(self):
    return self.state

  def getParent(self):
    return self.parent

  def getDirection(self):
    return self.direction

class node2(node):
  def __init__(self, parent, direction, state, position, foodLeft):
    self.parent = parent
    self.direction = direction
    self.state = state
    self.position = position
    self.foodLeft = foodLeft

  def getPosition(self):
    return self.position

  def getFoodLeft(self):
    return self.foodLeft

class node3(node2):
  def __init__(self, position, parent):
    self.position = position
    self.parent = parent


